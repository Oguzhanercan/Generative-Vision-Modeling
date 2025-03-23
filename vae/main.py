# main.py
import torch
import torch.nn as nn
import yaml
import logging
import lpips
import argparse  # Added for command-line argument parsing
import os  # Added for os.path in checkpointing
from pytorch_msssim import SSIM
from data.dataloader import get_dataloaders
from models.vae import VAE
from utils.logger import setup_workdir_and_logging, log_validation_epoch
from core.trainer import train_epoch
from core.validator import validate

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a Variational Autoencoder (VAE) for generative vision modeling.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration YAML file (default: config/config.yaml)"
    )
    return parser.parse_args()

def load_config(config_path):
    """
    Load and parse the configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Model parameters
    config["model"]["model_type"] = config["model"].get("model_type", "vae")  # Default to "vae"
    config["model"]["input_dim"] = int(config["model"]["input_dim"])
    config["model"]["latent_dim"] = int(config["model"]["latent_dim"])
    config["model"]["vq_embedding_dim"] = int(config["model"]["vq_embedding_dim"])
    config["model"]["vq_num_embeddings"] = int(config["model"]["vq_num_embeddings"])
    config["model"]["input_size"] = int(config["model"]["input_size"])
    config["model"]["vq_beta"] = float(config["model"]["vq_beta"])
    config["model"]["ema_decay"] = float(config["model"].get("ema_decay", 0.99))
    config["model"]["constant_variance"] = bool(config["model"].get("constant_variance", False))
    config["model"]["downsample_positions"] = [int(x) for x in config["model"].get("downsample_positions", [])]
    config["model"]["block_configs"] = config["model"].get("block_configs", [{"type": "resnet"}])
    
    # Training parameters
    config["training"]["learning_rate"] = float(config["training"]["learning_rate"])
    config["training"]["epochs"] = int(config["training"]["epochs"])
    config["training"]["batch_size"] = int(config["training"]["batch_size"])
    config["training"]["log_interval"] = int(config["training"]["log_interval"])
    config["training"]["limit_samples"] = int(config["training"]["limit_samples"])
    config["training"]["n_best"] = int(config["training"]["n_best"])
    config["training"]["output_dir"] = config["training"].get("output_dir", "runs")
    
    # Data parameters
    config["data"]["image_size"] = int(config["data"]["image_size"])
    config["data"]["val_split"] = float(config["data"]["val_split"])
    
    # Loss parameters
    config["loss"]["use_perceptual"] = bool(config["loss"].get("use_perceptual", False))
    config["loss"]["perceptual_weight"] = float(config["loss"].get("perceptual_weight", 1.0))
    config["loss"]["use_lpips"] = bool(config["loss"].get("use_lpips", False))
    config["loss"]["lpips_weight"] = float(config["loss"].get("lpips_weight", 1.0))
    config["loss"]["use_ssim"] = bool(config["loss"].get("use_ssim", False))
    config["loss"]["ssim_weight"] = float(config["loss"].get("ssim_weight", 1.0))
    config["loss"]["beta_recon"] = float(config["loss"].get("beta_recon", 0.5))
    config["loss"]["kl_annealing_epochs"] = int(config["loss"].get("kl_annealing_epochs", 20))
    config["loss"]["beta_kl"] = float(config["loss"].get("beta_kl", 1.0))
    config["loss"]["reconstruction_loss"] = config["loss"].get("reconstruction_loss", "mse")

    # Memory optimization parameters
    config["optimize_memory_usage"] = config.get("optimize_memory_usage", {})
    config["optimize_memory_usage"]["use_gradient_checkpointing"] = bool(
        config["optimize_memory_usage"].get("use_gradient_checkpointing", False)
    )
    config["optimize_memory_usage"]["use_mixed_precision"] = bool(
        config["optimize_memory_usage"].get("use_mixed_precision", False)
    )
    config["optimize_memory_usage"]["perceptual_loss_batch_size"] = int(
        config["optimize_memory_usage"].get("perceptual_loss_batch_size", 0)
    )

    return config

def main(config_path):
    """
    Main function to orchestrate the training process.

    Args:
        config_path (str): Path to the configuration file.
    """
    config = load_config(config_path)
    device = torch.device(config["training"]["device"])
    
    # Setup logging and visualizations
    workdir, writer, tensorboard_dir, checkpoint_dir = setup_workdir_and_logging(config)
    logging.info("Starting training with configuration: %s", config)

    # Load data
    train_loader, val_loader = get_dataloaders(config)
    fixed_train_batch = next(iter(train_loader))[0][:8].to(device)
    fixed_val_batch = next(iter(val_loader))[0][:8].to(device)

    # Initialize models and optimizers
    vae = VAE(config).to(device)
    vgg = nn.Sequential(*list(torch.hub.load('pytorch/vision', 'vgg16', pretrained=True).features)).to(device).eval()
    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device) if config["loss"]["use_lpips"] else None
    ssim_loss_fn = SSIM(data_range=1.0, size_average=True, channel=config["data"]["data_channel"]).to(device) if config["loss"]["use_ssim"] else None
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=config["training"]["learning_rate"])

    # Training loop
    global_step = 0
    for epoch in range(config["training"]["epochs"]):
        total_vae_loss, global_step = train_epoch(
            vae=vae,
            train_loader=train_loader,
            vgg=vgg,
            config=config,
            epoch=epoch,
            vae_optimizer=vae_optimizer,
            lpips_loss_fn=lpips_loss_fn,
            ssim_loss_fn=ssim_loss_fn,
            device=device,
            fixed_train_batch=fixed_train_batch,
            writer=writer,
            global_step=global_step,
            use_gradient_checkpointing=config["optimize_memory_usage"]["use_gradient_checkpointing"],
            use_mixed_precision=config["optimize_memory_usage"]["use_mixed_precision"],
            perceptual_loss_batch_size=config["optimize_memory_usage"]["perceptual_loss_batch_size"]
        )

        # Validation
        val_metrics = validate(
        vae=vae,
        val_loader=val_loader,
        vgg=vgg,
        config=config,
        epoch=epoch,
        lpips_loss_fn=lpips_loss_fn,
        ssim_loss_fn=ssim_loss_fn,
        device=device,
        fixed_val_batch=fixed_val_batch,
        use_gradient_checkpointing=config["optimize_memory_usage"]["use_gradient_checkpointing"],
        use_mixed_precision=config["optimize_memory_usage"]["use_mixed_precision"],
        perceptual_loss_batch_size=config["optimize_memory_usage"]["perceptual_loss_batch_size"]
    )
        val_vae_loss, val_recon, val_klg, val_kli, val_percep, val_lpips, val_ssim, val_vq, val_recon_first, val_x_first = val_metrics
        
        # Log validation metrics
        log_validation_epoch(writer, epoch, total_vae_loss, train_loader, val_vae_loss, val_recon, val_klg, val_kli, 
                            val_percep, val_lpips, val_ssim, val_vq, val_recon_first, val_x_first)

        

        # Checkpointing
        if epoch % 10 == 0 and epoch > 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': vae_optimizer.state_dict(),
                'loss': total_vae_loss/len(train_loader),
            }, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")

    writer.close()
    logging.info("Training completed.")

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    # Pass the config path to main
    main(args.config)