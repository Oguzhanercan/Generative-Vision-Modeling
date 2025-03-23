# losses/vae_loss.py
import torch
import torch.nn.functional as F
def compute_vae_loss(vae, vgg, config, epoch, x, recon, mu_or_z, logvar, z, vq_loss, lpips_loss_fn, ssim_loss_fn):
    """
    Compute VAE loss using precomputed VAE outputs.
    
    Args:
        vae: VAE model (for reference, not used for forward pass here)
        vgg: VGG model for perceptual loss
        config: Configuration dictionary
        epoch: Current epoch
        x: Input tensor
        recon: Reconstructed output from VAE
        mu: Mean from encoder
        logvar: Log variance from encoder
        z: Latent representation
        vq_loss: Vector quantization loss (0 if not used)
        lpips_loss_fn: LPIPS loss function (None if not used)
        ssim_loss_fn: SSIM loss function (None if not used)
    
    Returns:
        Tuple of (vae_loss, recon_loss, klg, kli, percep_loss, lpips_loss, ssim_loss, vq_loss, recon)
    """
    # Reconstruction loss
    recon_loss_type = config["loss"]["reconstruction_loss"]
    if recon_loss_type == "mse":
        recon_loss = config["loss"]["beta_recon"] * F.mse_loss(recon, x)
    elif recon_loss_type == "l1":
        recon_loss = config["loss"]["beta_recon"] * F.l1_loss(recon, x)
    else:
        raise ValueError(f"Unknown reconstruction_loss type: {recon_loss_type}")

    # KL divergence (only for VAE)
    kl_loss = torch.tensor(0.0, device=x.device)
    if config["model"]["model_type"] == "vae" and logvar is not None:
        kl_loss = -0.5 * torch.mean(1 + logvar - mu_or_z.pow(2) - logvar.exp())
        kl_loss = config["loss"]["beta_kl"] * kl_loss

    # Perceptual, LPIPS, SSIM losses
    percep_loss = torch.tensor(0.0, device=x.device)
    lpips_loss = torch.tensor(0.0, device=x.device)
    ssim_loss = torch.tensor(0.0, device=x.device)

    if config["loss"]["use_perceptual"] and vgg is not None:
        perceptual_loss_batch_size = config["optimize_memory_usage"]["perceptual_loss_batch_size"]
        if perceptual_loss_batch_size > 0:
            batch_size = x.size(0)
            percep_loss = 0.0
            for i in range(0, batch_size, perceptual_loss_batch_size):
                x_batch = x[i:i + perceptual_loss_batch_size]
                recon_batch = recon[i:i + perceptual_loss_batch_size]
                percep_loss += F.mse_loss(vgg(x_batch), vgg(recon_batch), reduction='mean')
            percep_loss = percep_loss / (batch_size // perceptual_loss_batch_size)
        else:
            percep_loss = F.mse_loss(vgg(x), vgg(recon), reduction='mean')
        percep_loss = config["loss"]["perceptual_weight"] * percep_loss
    if config["loss"]["use_lpips"] and lpips_loss_fn is not None:
        lpips_loss = config["loss"]["lpips_weight"] * lpips_loss_fn(recon, x).mean()
    if config["loss"]["use_ssim"] and ssim_loss_fn is not None:
        ssim_loss = config["loss"]["ssim_weight"] * (1 - ssim_loss_fn(recon, x))

    # Total loss
    vae_loss = recon_loss + kl_loss + vq_loss + percep_loss + lpips_loss + ssim_loss

    # Split KL for logging
    klg = kl_loss if config["model"]["model_type"] == "vae" else torch.tensor(0.0, device=x.device)
    kli = torch.tensor(0.0, device=x.device)

    return vae_loss, recon_loss, klg, kli, percep_loss, lpips_loss, ssim_loss, vq_loss, recon