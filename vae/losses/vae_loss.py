# losses/vae_loss.py
import torch
import torch.nn.functional as F

def compute_vae_loss(vae, vgg, config, epoch, x, lpips_loss_fn=None, ssim_loss_fn=None):
    recon, mu, logvar, z, vq_loss = vae(x)
    latent_dims = mu.numel() // mu.shape[0]
    
    # KL terms
    mu_global = mu.mean(dim=0, keepdim=True)
    sigma_global = ((z - mu_global)**2).mean(dim=0, keepdim=True)
    klg = 0.5 * (sigma_global + mu_global**2 - 1 - torch.log(sigma_global + 1e-8)).sum() / latent_dims
    kli = 0.5 * (torch.exp(logvar) - 1 - logvar).sum(dim=(1,2,3)).mean() / latent_dims
    
    # Reconstruction loss
    if config["loss"]["reconstruction_loss"] == "mse":
        recon_loss = config["loss"]["beta_recon"] * F.mse_loss(recon, x)
    else:  # "l1"
        recon_loss = config["loss"]["beta_recon"] * F.l1_loss(recon, x)
    
    # Optional perceptual losses
    percep_loss = lpips_loss = ssim_loss = torch.tensor(0.0, device=x.device)
    if config["loss"]["use_perceptual"]:
        vgg_input = F.interpolate(x.clamp(0, 1), size=224, mode='bilinear')
        vgg_recon = F.interpolate(recon.clamp(0, 1), size=224, mode='bilinear')
        percep_loss = config["loss"]["perceptual_weight"] * F.l1_loss(vgg(vgg_input), vgg(vgg_recon))
    if config["loss"]["use_lpips"] and lpips_loss_fn:
        lpips_loss = config["loss"]["lpips_weight"] * lpips_loss_fn(recon.clamp(0, 1), x.clamp(0, 1)).mean()
    if config["loss"]["use_ssim"] and ssim_loss_fn:
        ssim_loss = config["loss"]["ssim_weight"] * (1 - ssim_loss_fn(recon.clamp(0, 1), x.clamp(0, 1)))
    
    # Total loss with conditional KLI and VQ
    kl_weight = min(1.0, epoch / config["loss"]["kl_annealing_epochs"])
    beta_kl = config["loss"]["beta_kl"]
    if config["model"]["constant_variance"]:
        vae_loss = recon_loss + kl_weight * (beta_kl * klg) + percep_loss + lpips_loss + ssim_loss + vq_loss
    else:
        vae_loss = recon_loss + kl_weight * (beta_kl * klg + beta_kl * kli) + percep_loss + lpips_loss + ssim_loss + vq_loss
    
    return vae_loss, recon_loss, klg, kli, percep_loss, lpips_loss, ssim_loss, vq_loss, recon