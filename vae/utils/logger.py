# utils/logger.py
import os
import logging
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from vis_tools.vis import visualize_architecture_graph, visualize_training_pipeline_graph, visualize_attn_block, visualize_resnet_block

# Global flag to track if input images have been logged
INPUT_IMAGES_LOGGED = False

def setup_logging(config, workdir):
    """
    Set up logging to file and console.

    Args:
        config (dict): Configuration dictionary.
        workdir (str): Working directory for logs.
    """
    log_file = os.path.join(workdir, "train.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.getLogger().handlers = []
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info("Logging setup complete.")

def setup_visualizations(config, workdir):
    """
    Generate and save visualization graphs.

    Args:
        config (dict): Configuration dictionary.
        workdir (str): Working directory for visualizations.
    """
    arch_graph_path = os.path.join(workdir, "vae_architecture.png")
    pipeline_graph_path = os.path.join(workdir, "vae_training_pipeline.png")
    attn_block_path = os.path.join(workdir, "attn_block.png")
    resnet_block_path = os.path.join(workdir, "resnet_block.png")
    
    visualize_architecture_graph(config, output_path=arch_graph_path)
    visualize_training_pipeline_graph(config, output_path=pipeline_graph_path)
    visualize_attn_block(output_path=attn_block_path)
    visualize_resnet_block(output_path=resnet_block_path)
    
    logging.info(f"Architecture graph saved to {arch_graph_path}")
    logging.info(f"Training pipeline graph saved to {pipeline_graph_path}")
    logging.info(f"Attention block graph saved to {attn_block_path}")
    logging.info(f"ResNet block graph saved to {resnet_block_path}")

def setup_workdir_and_logging(config):
    """
    Set up the working directory, logging, and visualizations.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        tuple: (workdir, writer, tensorboard_dir, checkpoint_dir)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workdir = os.path.join(config["training"]["output_dir"], f"run_{timestamp}")
    os.makedirs(workdir, exist_ok=True)
    
    tensorboard_dir = os.path.join(workdir, "tensorboard")
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    config_file_path = os.path.join(workdir, "config.yaml")
    with open(config_file_path, 'w') as f:
        import yaml
        yaml.safe_dump(config, f)
    logging.info(f"Configuration saved to {config_file_path}")
    
    setup_logging(config, workdir)
    setup_visualizations(config, workdir)
    
    writer = SummaryWriter(log_dir=tensorboard_dir)
    return workdir, writer, tensorboard_dir, checkpoint_dir

def log_training_step(writer, global_step, epoch, config, avg_vae_loss, recon_loss_val, klg_val, kli_val, 
                      percep_loss_val, lpips_loss_val, ssim_loss_val, vq_loss_val, train_recon_first=None, train_input_first=None):
    """
    Log training metrics and reconstructed images to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter.
        global_step (int): Global step number.
        epoch (int): Current epoch.
        config (dict): Configuration dictionary.
        avg_vae_loss (float): Average VAE loss.
        recon_loss_val (float): Reconstruction loss.
        klg_val (float): KL divergence (global).
        kli_val (float): KL divergence (individual).
        percep_loss_val (float): Perceptual loss.
        lpips_loss_val (float): LPIPS loss.
        ssim_loss_val (float): SSIM loss.
        vq_loss_val (float): VQ loss.
        train_recon_first (torch.Tensor, optional): Reconstructed images.
        train_input_first (torch.Tensor, optional): Input images.
    """
    global INPUT_IMAGES_LOGGED

    log_message = (
        f"Step {global_step} (Epoch {epoch+1}/{config['training']['epochs']}) - "
        f"Train VAE Loss: {avg_vae_loss:.4f} - Recon: {recon_loss_val:.4f} - "
        f"KLG: {klg_val:.4f} - KLI: {kli_val:.4f} - "
        f"Percep: {percep_loss_val:.4f} - LPIPS: {lpips_loss_val:.4f} - "
        f"SSIM: {ssim_loss_val:.4f} - VQ: {vq_loss_val:.4f}"
    )
    logging.info(log_message)

    writer.add_scalar("Loss/Train/VAE", avg_vae_loss, global_step)
    writer.add_scalar("Loss/Train/Recon", recon_loss_val, global_step)
    writer.add_scalar("Loss/Train/KLG", klg_val, global_step)
    writer.add_scalar("Loss/Train/KLI", kli_val, global_step)
    writer.add_scalar("Loss/Train/Percep", percep_loss_val, global_step)
    writer.add_scalar("Loss/Train/LPIPS", lpips_loss_val, global_step)
    writer.add_scalar("Loss/Train/SSIM", ssim_loss_val, global_step)
    writer.add_scalar("Loss/Train/VQ", vq_loss_val, global_step)

    # Log input images only once at the start of training
    if not INPUT_IMAGES_LOGGED and train_input_first is not None:
        num_images = min(8, train_input_first.size(0))
        train_input_first = torch.clamp(train_input_first, 0, 1)
        train_input_grid = make_grid(train_input_first.detach().cpu(), nrow=4, normalize=False).unsqueeze(0)
        writer.add_images("Train/Input", train_input_grid, 0)  # Log at step 0
        INPUT_IMAGES_LOGGED = True

    # Log reconstructions every time this function is called with train_recon_first
    if train_recon_first is not None:
        num_images = min(8, train_recon_first.size(0))
        train_recon_first = torch.clamp(train_recon_first, 0, 1)
        train_recon_grid = make_grid(train_recon_first.detach().cpu(), nrow=4, normalize=False).unsqueeze(0)
        writer.add_images("Train/Recon", train_recon_grid, global_step)
    writer.flush()

def log_validation_epoch(writer, epoch, total_vae_loss, train_loader, val_vae_loss, val_recon, val_klg, val_kli, 
                         val_percep, val_lpips, val_ssim, val_vq, val_recon_first, val_input_first=None):
    """
    Log validation metrics and reconstructed images to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter.
        epoch (int): Current epoch.
        total_vae_loss (float): Total training VAE loss.
        train_loader: Training DataLoader.
        val_vae_loss (float): Validation VAE loss.
        val_recon (float): Validation reconstruction loss.
        val_klg (float): Validation KL divergence (global).
        val_kli (float): Validation KL divergence (individual).
        val_percep (float): Validation perceptual loss.
        val_lpips (float): Validation LPIPS loss.
        val_ssim (float): Validation SSIM loss.
        val_vq (float): Validation VQ loss.
        val_recon_first (torch.Tensor): Validation reconstructed images.
        val_input_first (torch.Tensor, optional): Validation input images.
    """
    global INPUT_IMAGES_LOGGED

    log_message = (
        f"Epoch {epoch+1} completed - Train VAE Loss: {total_vae_loss/len(train_loader):.4f} - "
        f"Val VAE Loss: {val_vae_loss:.4f} - Val Recon: {val_recon:.4f} - "
        f"Val KLG: {val_klg:.4f} - Val KLI: {val_kli:.4f} - "
        f"Val Percep: {val_percep:.4f} - Val LPIPS: {val_lpips:.4f} - "
        f"Val SSIM: {val_ssim:.4f} - Val VQ: {val_vq:.4f}"
    )
    logging.info(log_message)

    writer.add_scalar("Loss/Val/VAE_Epoch", val_vae_loss, epoch)
    writer.add_scalar("Loss/Val/Recon", val_recon, epoch)
    writer.add_scalar("Loss/Val/KLG", val_klg, epoch)
    writer.add_scalar("Loss/Val/KLI", val_kli, epoch)
    writer.add_scalar("Loss/Val/Percep", val_percep, epoch)
    writer.add_scalar("Loss/Val/LPIPS", val_lpips, epoch)
    writer.add_scalar("Loss/Val/SSIM", val_ssim, epoch)
    writer.add_scalar("Loss/Val/VQ", val_vq, epoch)

    # Log input images only once at the start of training
    if not INPUT_IMAGES_LOGGED and val_input_first is not None:
        num_images = min(8, val_input_first.size(0))
        val_input_first = torch.clamp(val_input_first, 0, 1)
        val_input_grid = make_grid(val_input_first.cpu(), nrow=4, normalize=False).unsqueeze(0)
        writer.add_images("Val/Input", val_input_grid, 0)  # Log at step 0
        INPUT_IMAGES_LOGGED = True

    # Log reconstructions every epoch
    if val_recon_first is not None:
        num_images = min(8, val_recon_first.size(0))
        val_recon_first = torch.clamp(val_recon_first, 0, 1)
        val_recon_grid = make_grid(val_recon_first.cpu(), nrow=4, normalize=False).unsqueeze(0)
        writer.add_images("Val/Recon", val_recon_grid, epoch)
    writer.flush()