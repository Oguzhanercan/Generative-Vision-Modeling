# core/validator.py
import torch
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast
from losses.vae_loss import compute_vae_loss

def validate(vae, val_loader, vgg, config, epoch, lpips_loss_fn, ssim_loss_fn, device, fixed_val_batch=None,
             use_gradient_checkpointing=False, use_mixed_precision=False, perceptual_loss_batch_size=0):
    """
    Validate the VAE model on the validation set.

    Args:
        vae: VAE model.
        val_loader: Validation DataLoader.
        vgg: VGG model for perceptual loss.
        config: Configuration dictionary.
        epoch: Current epoch.
        lpips_loss_fn: LPIPS loss function.
        ssim_loss_fn: SSIM loss function.
        device: Device to run the model on.
        fixed_val_batch: Fixed validation batch for consistent visualization.
        use_gradient_checkpointing: Whether to use gradient checkpointing.
        use_mixed_precision: Whether to use mixed precision training.
        perceptual_loss_batch_size: Batch size for perceptual loss computation (0 to disable).

    Returns:
        tuple: Validation metrics and images.
    """
    vae.eval()
    total_vae_loss = 0
    recon_first, x_first = None, None
    with torch.no_grad():
        for i, (x, _) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")):
            x = x.to(device)
            if use_mixed_precision:
                with autocast():
                    if use_gradient_checkpointing:
                        recon, mu_or_z, logvar, z, vq_loss, _ = checkpoint(vae, x, use_reentrant=False)
                    else:
                        recon, mu_or_z, logvar, z, vq_loss, _ = vae(x)
                    vae_loss, recon_loss, klg, kli, percep_loss, lpips_loss, ssim_loss, vq_loss, recon = compute_vae_loss(
                        vae, vgg, config, epoch, x, recon, mu_or_z, logvar, z, vq_loss, lpips_loss_fn, ssim_loss_fn
                    )
            else:
                if use_gradient_checkpointing:
                    recon, mu_or_z, logvar, z, vq_loss, _ = checkpoint(vae, x, use_reentrant=False)
                else:
                    recon, mu_or_z, logvar, z, vq_loss, _ = vae(x)
                vae_loss, recon_loss, klg, kli, percep_loss, lpips_loss, ssim_loss, vq_loss, recon = compute_vae_loss(
                    vae, vgg, config, epoch, x, recon, mu_or_z, logvar, z, vq_loss, lpips_loss_fn, ssim_loss_fn
                )
            total_vae_loss += vae_loss.item()
            if i == 0 and fixed_val_batch is None:
                recon_first = recon
                x_first = x
        
        if fixed_val_batch is not None:
            x = fixed_val_batch.to(device)
            if use_mixed_precision:
                with autocast():
                    if use_gradient_checkpointing:
                        recon, mu_or_z, logvar, z, vq_loss, _ = checkpoint(vae, x, use_reentrant=False)
                    else:
                        recon, mu_or_z, logvar, z, vq_loss, _ = vae(x)
                    vae_loss, recon_loss, klg, kli, percep_loss, lpips_loss, ssim_loss, vq_loss, recon = compute_vae_loss(
                        vae, vgg, config, epoch, x, recon, mu_or_z, logvar, z, vq_loss, lpips_loss_fn, ssim_loss_fn
                    )
            else:
                if use_gradient_checkpointing:
                    recon, mu_or_z, logvar, z, vq_loss, _ = checkpoint(vae, x, use_reentrant=False)
                else:
                    recon, mu_or_z, logvar, z, vq_loss, _ = vae(x)
                vae_loss, recon_loss, klg, kli, percep_loss, lpips_loss, ssim_loss, vq_loss, recon = compute_vae_loss(
                    vae, vgg, config, epoch, x, recon, mu_or_z, logvar, z, vq_loss, lpips_loss_fn, ssim_loss_fn
                )
            recon_first = recon
            x_first = x
    
    avg_vae_loss = total_vae_loss / len(val_loader)
    return (avg_vae_loss, recon_loss.item(), klg.item(), kli.item(), percep_loss.item(), 
            lpips_loss.item(), ssim_loss.item(), vq_loss.item(), recon_first, x_first)