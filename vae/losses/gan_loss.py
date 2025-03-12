# losses/gan_loss.py
import torch

def compute_gan_loss(discriminator, real, fake, config):
    real_logits = discriminator(real)
    fake_logits = discriminator(fake)
    real_loss = torch.mean(torch.relu(1.0 - real_logits))  # WGAN-like
    fake_loss = torch.mean(torch.relu(1.0 + fake_logits))
    d_loss = real_loss + fake_loss
    g_loss = -torch.mean(fake_logits) * config["gan"]["gan_weight"]
    return d_loss, g_loss