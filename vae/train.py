# train.py
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from models.vae import VAE
from models.discriminator import Discriminator
from losses.vae_loss import compute_vae_loss
from losses.gan_loss import compute_gan_loss
from data.dataloader import get_dataloaders
from utils.helpers import setup_logging
import lpips
from pytorch_msssim import SSIM
import os
from torchvision.utils import make_grid
from heapq import heappush, heapreplace
from datetime import datetime

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config["training"]["learning_rate"] = float(config["training"]["learning_rate"])
    config["training"]["epochs"] = int(config["training"]["epochs"])
    config["training"]["batch_size"] = int(config["training"]["batch_size"])
    config["training"]["log_interval"] = int(config["training"]["log_interval"])
    config["training"]["limit_samples"] = int(config["training"]["limit_samples"])
    config["training"]["n_best"] = int(config["training"]["n_best"])
    config["data"]["image_size"] = int(config["data"]["image_size"])
    config["data"]["val_split"] = float(config["data"]["val_split"])
    config["model"]["input_dim"] = int(config["model"]["input_dim"])
    config["model"]["latent_dim"] = int(config["model"]["latent_dim"])
    config["loss"]["kl_annealing_epochs"] = int(config["loss"]["kl_annealing_epochs"])
    config["loss"]["beta_kl"] = float(config["loss"]["beta_kl"])
    config["loss"]["beta_recon"] = float(config["loss"]["beta_recon"])
    config["loss"]["perceptual_weight"] = float(config["loss"]["perceptual_weight"])
    config["loss"]["lpips_weight"] = float(config["loss"]["lpips_weight"])
    config["loss"]["ssim_weight"] = float(config["loss"]["ssim_weight"])
    if config["gan"]["enabled"]:
        config["gan"]["discriminator_lr"] = float(config["gan"]["discriminator_lr"])
        config["gan"]["gan_weight"] = float(config["gan"]["gan_weight"])
    if config["model"]["use_vq"]:
        config["model"]["vq_num_embeddings"] = int(config["model"]["vq_num_embeddings"])
        config["model"]["vq_embedding_dim"] = int(config["model"]["vq_embedding_dim"])
        config["model"]["vq_beta"] = float(config["model"]["vq_beta"])
    return config

def validate(vae, val_loader, vgg, config, epoch, lpips_loss_fn, ssim_loss_fn, device, fixed_val_batch=None):
    vae.eval()
    total_vae_loss = 0
    recon_first, x_first = None, None
    with torch.no_grad():
        for i, (x, _) in enumerate(val_loader):
            x = x.to(device)
            vae_loss, recon_loss, klg, kli, percep_loss, lpips_loss, ssim_loss, vq_loss, recon = compute_vae_loss(
                vae, vgg, config, epoch, x, lpips_loss_fn, ssim_loss_fn
            )
            total_vae_loss += vae_loss.item()
            if i == 0 and fixed_val_batch is None:  # Initial setup
                recon_first = recon
                x_first = x
        # Use fixed validation batch if provided
        if fixed_val_batch is not None:
            x = fixed_val_batch.to(device)
            vae_loss, recon_loss, klg, kli, percep_loss, lpips_loss, ssim_loss, vq_loss, recon = compute_vae_loss(
                vae, vgg, config, epoch, x, lpips_loss_fn, ssim_loss_fn
            )
            recon_first = recon
            x_first = x
    avg_vae_loss = total_vae_loss / len(val_loader)
    return (avg_vae_loss, recon_loss.item(), klg.item(), kli.item(), percep_loss.item(), 
            lpips_loss.item(), ssim_loss.item(), vq_loss.item(), recon_first, x_first)

def train(config):
    device = torch.device(config["training"]["device"])
    
    # Create unique workdir with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workdir = os.path.join(config["training"]["output_dir"], f"run_{timestamp}")
    os.makedirs(workdir, exist_ok=True)
    
    # Update paths
    log_file = os.path.join(workdir, "train.log")
    tensorboard_dir = os.path.join(workdir, "tensorboard")
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup logging and TensorBoard
    setup_logging(config, log_file_path=log_file)  # Updated argument name
    writer = SummaryWriter(tensorboard_dir)
    
    # Data
    train_loader, val_loader = get_dataloaders(config)
    
    # Fix samples for consistent visualization
    fixed_train_batch = next(iter(train_loader))[0][:8]  # First 8 samples from train
    fixed_val_batch = next(iter(val_loader))[0][:8]      # First 8 samples from val
    
    # Models
    vae = VAE(
        config["model"]["input_dim"],
        config["model"]["hidden_dims"],
        config["model"]["latent_dim"],
        config["model"]["constant_variance"],
        input_size=config["data"]["image_size"],
        use_vq=config["model"]["use_vq"],
        vq_num_embeddings=config["model"]["vq_num_embeddings"],
        vq_embedding_dim=config["model"]["vq_embedding_dim"],
        vq_beta=config["model"]["vq_beta"]
    ).to(device)
    vgg = nn.Sequential(*list(torch.hub.load('pytorch/vision', 'vgg16', pretrained=True).features)).to(device).eval()
    discriminator = None
    if config["gan"]["enabled"]:
        discriminator = Discriminator(config["model"]["input_dim"], config["model"]["hidden_dims"]).to(device)
    
    # Loss functions
    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device) if config["loss"]["use_lpips"] else None
    ssim_loss_fn = SSIM(data_range=1.0, size_average=True, channel=config["model"]["input_dim"]).to(device) \
        if config["loss"]["use_ssim"] else None
    
    # Optimizers
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=config["training"]["learning_rate"])
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config["gan"]["discriminator_lr"]) \
        if config["gan"]["enabled"] else None
    
    # Track n best models (min-heap of (val_loss, epoch, path))
    n_best = config["training"]["n_best"]
    best_models = []
    
    # Global step counter
    global_step = 0
    
    # Training loop
    for epoch in range(config["training"]["epochs"]):
        vae.train()
        if discriminator:
            discriminator.train()
        
        total_vae_loss = 0
        for i, (x, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")):
            x = x.to(device)
            
            vae_optimizer.zero_grad()
            recon, mu, logvar, z, vq_loss = vae(x)
            vae_loss, recon_loss, klg, kli, percep_loss, lpips_loss, ssim_loss, vq_loss, recon = compute_vae_loss(
                vae, vgg, config, epoch, x, lpips_loss_fn, ssim_loss_fn
            )
            total_vae_loss += vae_loss.item()
            
            if config["gan"]["enabled"]:
                disc_optimizer.zero_grad()
                recon, _, _, _, _ = vae(x)
                d_loss, g_loss = compute_gan_loss(discriminator, x, recon, config)
                d_loss.backward()
                disc_optimizer.step()
                vae_loss += g_loss
            
            vae_loss.backward()
            vae_optimizer.step()
            
            global_step += 1
            
            # Training logging per step (no validation)
            if global_step % config["training"]["log_interval"] == 0:
                avg_vae_loss = total_vae_loss / (i + 1)
                
                # Compute reconstructions for fixed train batch
                with torch.no_grad():
                    _, recon_loss, klg, kli, percep_loss, lpips_loss, ssim_loss, vq_loss, train_recon_first = compute_vae_loss(
                        vae, vgg, config, epoch, fixed_train_batch.to(device), lpips_loss_fn, ssim_loss_fn
                    )
                train_x_first = fixed_train_batch
                
                log_message = (
                    f"Step {global_step} (Epoch {epoch+1}/{config['training']['epochs']}) - "
                    f"Train VAE Loss: {avg_vae_loss:.4f} - "
                    f"Recon: {recon_loss.item():.4f} - "
                    f"KLG: {klg.item():.4f} - KLI: {kli.item():.4f} - "
                    f"Percep: {percep_loss.item():.4f} - "
                    f"LPIPS: {lpips_loss.item():.4f} - "
                    f"SSIM: {ssim_loss.item():.4f} - "
                    f"VQ: {vq_loss.item():.4f}"
                )
                logging.info(log_message)
                
                # Scalar logging (train only)
                writer.add_scalar("Loss/Train/VAE", avg_vae_loss, global_step)
                writer.add_scalar("Loss/Train/Recon", recon_loss.item(), global_step)
                writer.add_scalar("Loss/Train/KLG", klg.item(), global_step)
                writer.add_scalar("Loss/Train/KLI", kli.item(), global_step)
                writer.add_scalar("Loss/Train/Percep", percep_loss.item(), global_step)
                writer.add_scalar("Loss/Train/LPIPS", lpips_loss.item(), global_step)
                writer.add_scalar("Loss/Train/SSIM", ssim_loss.item(), global_step)
                writer.add_scalar("Loss/Train/VQ", vq_loss.item(), global_step)
                
                # Visualization logging (train only)
                num_images = min(8, train_x_first.size(0))
                train_grid = make_grid(train_x_first[:num_images], nrow=4, normalize=True).unsqueeze(0)
                train_recon_grid = make_grid(train_recon_first[:num_images], nrow=4, normalize=True).unsqueeze(0)
                
                writer.add_images("Train/Input", train_grid, global_step)
                writer.add_images("Train/Recon", train_recon_grid, global_step)
        
        # Validation at epoch end
        val_metrics = validate(vae, val_loader, vgg, config, epoch, lpips_loss_fn, ssim_loss_fn, device, fixed_val_batch)
        val_vae_loss, val_recon, val_klg, val_kli, val_percep, val_lpips, val_ssim, val_vq, val_recon_first, val_x_first = val_metrics
        avg_vae_loss = total_vae_loss / len(train_loader)
        
        # Epoch-end logging
        log_message = (
            f"Epoch {epoch+1} completed - Train VAE Loss: {avg_vae_loss:.4f} - Val VAE Loss: {val_vae_loss:.4f} - "
            f"Val Recon: {val_recon:.4f} - Val KLG: {val_klg:.4f} - Val KLI: {val_kli:.4f} - "
            f"Val Percep: {val_percep:.4f} - Val LPIPS: {val_lpips:.4f} - Val SSIM: {val_ssim:.4f} - Val VQ: {val_vq:.4f}"
        )
        logging.info(log_message)
        
        # Scalar logging (epoch-end)
        writer.add_scalar("Loss/Train/VAE_Epoch", avg_vae_loss, epoch)
        writer.add_scalar("Loss/Val/VAE_Epoch", val_vae_loss, epoch)
        writer.add_scalar("Loss/Val/Recon", val_recon, epoch)
        writer.add_scalar("Loss/Val/KLG", val_klg, epoch)
        writer.add_scalar("Loss/Val/KLI", val_kli, epoch)
        writer.add_scalar("Loss/Val/Percep", val_percep, epoch)
        writer.add_scalar("Loss/Val/LPIPS", val_lpips, epoch)
        writer.add_scalar("Loss/Val/SSIM", val_ssim, epoch)
        writer.add_scalar("Loss/Val/VQ", val_vq, epoch)
        
        # Visualization logging (val at epoch end)
        num_images = min(8, val_x_first.size(0))
        val_grid = make_grid(val_x_first[:num_images], nrow=4, normalize=True).unsqueeze(0)
        val_recon_grid = make_grid(val_recon_first[:num_images], nrow=4, normalize=True).unsqueeze(0)
        
        writer.add_images("Val/Input", val_grid, epoch)
        writer.add_images("Val/Recon", val_recon_grid, epoch)
        
        # Save n best models (at epoch end)
        checkpoint_path = os.path.join(checkpoint_dir, f"vae_epoch_{epoch+1}_valloss_{val_vae_loss:.4f}.pth")
        torch.save(vae.state_dict(), checkpoint_path)
        if len(best_models) < n_best:
            heappush(best_models, (val_vae_loss, epoch, checkpoint_path))
        elif val_vae_loss < best_models[0][0]:
            old_path = heapreplace(best_models, (val_vae_loss, epoch, checkpoint_path))[2]
            if os.path.exists(old_path):
                os.remove(old_path)
    
    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    config = load_config()
    train(config)