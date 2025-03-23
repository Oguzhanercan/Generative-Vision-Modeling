# core/trainer.py
import torch
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler
from losses.vae_loss import compute_vae_loss
from core.schedulers import vq_beta_scheduler
from utils.logger import log_training_step

def train_epoch(vae, train_loader, vgg, config, epoch, vae_optimizer, lpips_loss_fn, ssim_loss_fn, device, 
                fixed_train_batch, writer, global_step, use_gradient_checkpointing=False, 
                use_mixed_precision=False, perceptual_loss_batch_size=0):
    """
    Train the VAE model for one epoch.

    Args:
        vae: VAE model.
        train_loader: Training DataLoader.
        vgg: VGG model for perceptual loss.
        config: Configuration dictionary.
        epoch: Current epoch.
        vae_optimizer: Optimizer for the VAE.
        lpips_loss_fn: LPIPS loss function.
        ssim_loss_fn: SSIM loss function.
        device: Device to run the model on.
        fixed_train_batch: Fixed training batch for consistent visualization.
        writer: TensorBoard SummaryWriter.
        global_step: Global step counter.
        use_gradient_checkpointing: Whether to use gradient checkpointing.
        use_mixed_precision: Whether to use mixed precision training.
        perceptual_loss_batch_size: Batch size for perceptual loss computation (0 to disable).

    Returns:
        tuple: (total_vae_loss, global_step)
    """
    vae.train()
    total_vae_loss = 0.0
    scaler = GradScaler(init_scale=2**10) if use_mixed_precision else None

    for i, (x, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")):
        x = x.to(device)
        vq_beta = vq_beta_scheduler(epoch, config)
        if vae.use_vq:
            vae.vq_layer.beta = vq_beta

        vae_optimizer.zero_grad(set_to_none=True)

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

        if use_mixed_precision:
            scaler.scale(vae_loss).backward()
            scaler.step(vae_optimizer)
            scaler.update()
        else:
            vae_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            vae_optimizer.step()

        if global_step % config["training"]["log_interval"] == 0:
            avg_vae_loss = total_vae_loss / (i + 1)
            with torch.no_grad():
                recon_loss_val = recon_loss.item()
                klg_val = klg.item()
                kli_val = kli.item()
                percep_loss_val = percep_loss.item()
                lpips_loss_val = lpips_loss.item()
                ssim_loss_val = ssim_loss.item()
                vq_loss_val = vq_loss.item()

                if use_mixed_precision:
                    with autocast():
                        recon_fixed, mu_or_z_fixed, logvar_fixed, z_fixed, vq_loss_fixed, _ = vae(fixed_train_batch)
                        _, _, _, _, _, _, _, _, train_recon_first = compute_vae_loss(
                            vae, vgg, config, epoch, fixed_train_batch, recon_fixed, mu_or_z_fixed, logvar_fixed, z_fixed, vq_loss_fixed, lpips_loss_fn, ssim_loss_fn
                        )
                else:
                    recon_fixed, mu_or_z_fixed, logvar_fixed, z_fixed, vq_loss_fixed, _ = vae(fixed_train_batch)
                    _, _, _, _, _, _, _, _, train_recon_first = compute_vae_loss(
                        vae, vgg, config, epoch, fixed_train_batch, recon_fixed, mu_or_z_fixed, logvar_fixed, z_fixed, vq_loss_fixed, lpips_loss_fn, ssim_loss_fn
                    )

                log_training_step(writer, global_step, epoch, config, avg_vae_loss, recon_loss_val, klg_val, kli_val, 
                                  percep_loss_val, lpips_loss_val, ssim_loss_val, vq_loss_val, train_recon_first, fixed_train_batch)

        del recon, mu_or_z, logvar, z, vq_loss, vae_loss, recon_loss, klg, kli, percep_loss, lpips_loss, ssim_loss
        torch.cuda.empty_cache()

        global_step += 1

    return total_vae_loss, global_step