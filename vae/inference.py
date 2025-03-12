# inference.py
import torch
import yaml
import argparse
from torchvision.utils import save_image, make_grid
from models.vae import VAE
from torch.utils.tensorboard import SummaryWriter
import os
from torchvision import datasets, transforms

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

def get_test_loader(config, test_path=None):
    """Create a test dataloader, using test_path if provided, else falling back to config dataset."""
    image_size = config["data"]["image_size"]
    batch_size = config["training"]["batch_size"]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    dataset_path = test_path if test_path else config["data"]["dataset_path"]
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

def inference(config, checkpoint_path, test_path=None, output_dir="inference_outputs", num_batches=5, tensorboard_dir=None):
    device = torch.device(config["training"]["device"])
    
    # Build model from config
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
    
    # Load checkpoint
    vae.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    vae.eval()
    
    # Test data
    test_loader = get_test_loader(config, test_path)
    
    # Output and TensorBoard setup
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir) if tensorboard_dir else None
    
    # Inference loop
    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            if i >= num_batches:
                break
            x = x.to(device)
            recon, mu, logvar, z, _ = vae(x)
            
            # Save images
            save_image(x, os.path.join(output_dir, f"input_batch_{i}.png"), nrow=4, normalize=True)
            save_image(recon, os.path.join(output_dir, f"recon_batch_{i}.png"), nrow=4, normalize=True)
            
            # TensorBoard logging
            if writer:
                num_images = min(8, x.size(0))
                input_grid = make_grid(x[:num_images], nrow=4, normalize=True)
                recon_grid = make_grid(recon[:num_images], nrow=4, normalize=True)
                writer.add_images("Inference/Input", input_grid, i)
                writer.add_images("Inference/Recon", recon_grid, i)
        
        # Sample from latent space (VAE) or codebook (VQ-VAE)
        if not config["model"]["use_vq"]:
            # Standard VAE: Sample from normal distribution
            z_sample = torch.randn(8, vae.latent_dim, config["data"]["image_size"] // (2 ** len(config["model"]["hidden_dims"])), 
                                   config["data"]["image_size"] // (2 ** len(config["model"]["hidden_dims"]))).to(device)
            samples = vae.decoder(z_sample)
        else:
            # VQ-VAE: Sample random codebook indices
            latent_h = config["data"]["image_size"] // (2 ** len(config["model"]["hidden_dims"]))
            z_sample_indices = torch.randint(0, config["model"]["vq_num_embeddings"], (8, latent_h, latent_h)).to(device)
            z_quantized = vae.vq_layer.embedding(z_sample_indices).permute(0, 3, 1, 2)
            samples = vae.decoder(z_quantized)
        
        # Save and log sampled images
        save_image(samples, os.path.join(output_dir, "sampled_images.png"), nrow=4, normalize=True)
        if writer:
            sample_grid = make_grid(samples, nrow=4, normalize=True)
            writer.add_images("Inference/Sampled", sample_grid, 0)
    
    if writer:
        writer.close()
    print(f"Inference completed. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with trained VAE/VQ-VAE model.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--test_path", default=None, help="Path to test dataset (optional)")
    parser.add_argument("--output_dir", default="inference_outputs", help="Directory for saving inference outputs")
    parser.add_argument("--num_batches", type=int, default=5, help="Number of batches to process")
    parser.add_argument("--tensorboard_dir", default=None, help="Directory for TensorBoard logs (optional)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    inference(config, args.checkpoint, args.test_path, args.output_dir, args.num_batches, args.tensorboard_dir)