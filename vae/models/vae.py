# vae.py
import torch
import torch.nn as nn
from einops import rearrange
from typing import List, Optional
from .blocks import ResnetBlock, AttnBlock, Downsample, Upsample
import torch.nn.functional as F

def swish(x):
    return x * torch.sigmoid(x)

class VectorQuantizer(nn.Module):
    # (Unchanged from your original code)
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25, use_ema: bool = False, decay: float = 0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.use_ema = use_ema
        self.decay = decay
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        if use_ema:
            self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
            self.register_buffer("ema_dw", torch.zeros_like(self.embedding.weight))

    def forward(self, z: torch.Tensor) -> tuple:
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)
        
        with torch.no_grad():
            distances = (z_flattened ** 2).sum(dim=1, keepdim=True) + \
                        (self.embedding.weight ** 2).sum(dim=1) - \
                        2 * torch.matmul(z_flattened, self.embedding.weight.t())
            encoding_indices = torch.argmin(distances, dim=1)
        
        z_q = self.embedding(encoding_indices).view(z.shape)
        
        if self.training and self.use_ema:
            with torch.no_grad():
                encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=z.device)
                encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
                encodings = encodings.view(z.shape[0], -1, self.num_embeddings)
                encodings_sum = encodings.sum(dim=(0, 1))
                
                dw = torch.matmul(encodings.permute(0, 2, 1), z.view(z.shape[0], -1, self.embedding_dim))
                dw = dw.sum(dim=0)
                
                self.ema_cluster_size.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
                self.ema_dw.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                
                n = self.ema_cluster_size.sum()
                cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
                updated_embedding = self.ema_dw / cluster_size.unsqueeze(1)
                self.embedding.weight.data.copy_(updated_embedding)
        
        commitment_loss = self.beta * F.mse_loss(z.detach(), z_q)
        embedding_loss = F.mse_loss(z, z_q.detach())
        vq_loss = commitment_loss + embedding_loss
        
        z_q = z + (z_q - z).detach()
        return z_q.permute(0, 3, 1, 2), vq_loss, encoding_indices

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config["model"]["input_dim"]
        self.hidden_dims = config["model"]["hidden_dims"]
        self.block_configs = config["model"]["block_configs"]
        self.downsample_positions = sorted(config["model"]["downsample_positions"])
        self.latent_dim = config["model"]["latent_dim"]
        self.num_groups = config["model"].get("num_groups", 32)
        self.model_type = config["model"]["model_type"]

        self.conv_in = nn.Conv2d(
            in_channels=config["data"]["data_channel"],
            out_channels=self.input_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.encoder_blocks = nn.ModuleList()
        current_dim = self.input_dim
        downsample_counter = 0

        for i in range(len(self.block_configs)):
            block_type = self.block_configs[i % len(self.block_configs)]["type"]
            out_dim = self.hidden_dims[min(i, len(self.hidden_dims)-1)]
            if block_type == "resnet":
                block = ResnetBlock(
                    in_channels=current_dim,
                    out_channels=out_dim,
                    num_groups=self.num_groups
                )
            elif block_type == "attention":
                block = AttnBlock(
                    in_channels=current_dim,
                    out_channels=out_dim,
                    num_groups=self.num_groups
                )
            else:
                raise ValueError(f"Unknown block type: {block_type}")
            self.encoder_blocks.append(block)
            current_dim = out_dim

            if downsample_counter < len(self.downsample_positions) and i == self.downsample_positions[downsample_counter]:
                self.encoder_blocks.append(Downsample(current_dim))
                downsample_counter += 1

        # Output convolution based on model type
        if self.model_type == "vae":
            self.conv_mu = nn.Conv2d(
                in_channels=current_dim,
                out_channels=self.latent_dim,
                kernel_size=1
            )
            self.conv_logvar = nn.Conv2d(
                in_channels=current_dim,
                out_channels=self.latent_dim,
                kernel_size=1
            )
        else:  # "ae"
            self.conv_out = nn.Conv2d(
                in_channels=current_dim,
                out_channels=self.latent_dim,
                kernel_size=1
            )

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.encoder_blocks:
            x = block(x)
        if self.model_type == "vae":
            mu = self.conv_mu(x)
            logvar = self.conv_logvar(x)
            return mu, logvar
        else:  # "ae"
            z = self.conv_out(x)
            return z, None  # Return None for logvar to maintain tuple consistency

class Decoder(nn.Module):
    # (Unchanged from your original code)
    def __init__(self, config):
        super().__init__()
        self.hidden_dims = config["model"]["hidden_dims"]
        self.block_configs = config["model"]["block_configs"]
        self.downsample_positions = sorted(config["model"]["downsample_positions"])
        self.latent_dim = config["model"]["latent_dim"]
        self.num_groups = config["model"].get("num_groups", 32)

        self.conv_in = nn.Conv2d(
            in_channels=self.latent_dim,
            out_channels=self.hidden_dims[-1],
            kernel_size=1
        )

        self.decoder_blocks = nn.ModuleList()
        current_dim = self.hidden_dims[-1]
        upsample_counter = len(self.downsample_positions) - 1

        for i in range(len(self.block_configs)):
            out_dim = self.hidden_dims[max(len(self.hidden_dims) - 1 - i, 0)]
            block_type = self.block_configs[i % len(self.block_configs)]["type"]
            if block_type == "resnet":
                block = ResnetBlock(
                    in_channels=current_dim,
                    out_channels=out_dim,
                    num_groups=self.num_groups
                )
            elif block_type == "attention":
                block = AttnBlock(
                    in_channels=current_dim,
                    out_channels=out_dim,
                    num_groups=self.num_groups
                )
            else:
                raise ValueError(f"Unknown block type: {block_type}")
            self.decoder_blocks.append(block)
            current_dim = out_dim

            if upsample_counter >= 0 and i == len(self.block_configs) - 1 - upsample_counter:
                self.decoder_blocks.append(Upsample(current_dim))
                upsample_counter -= 1

        self.conv_out = nn.Conv2d(
            in_channels=current_dim,
            out_channels=config["data"]["data_channel"],
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.final_act = nn.Sigmoid()

    def forward(self, z):
        x = self.conv_in(z)
        for block in self.decoder_blocks:
            x = block(x)
        #x = self.conv_out(x)
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()  # Cast to float32
            x = self.conv_out(x)
            x = self.final_act(x)
        #x = self.final_act(x)
        return x

class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.use_vq = config["model"]["use_vq"]
        self.model_type = config["model"]["model_type"]
        self.latent_dim = config["model"]["latent_dim"]
        self.constant_variance = config["model"]["constant_variance"]

        if self.use_vq:
            self.vq_layer = VectorQuantizer(
                num_embeddings=config["model"]["vq_num_embeddings"],
                embedding_dim=config["model"]["vq_embedding_dim"],
                beta=config["model"]["vq_beta"],
                use_ema=config["model"]["use_ema"],
                ema_decay=config["model"]["ema_decay"]
            )

    def reparameterize(self, mu, logvar):
        if self.model_type == "vae" and self.training and not self.constant_variance:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        # Encoder returns (mu, logvar) for VAE or (z, None) for AE
        encoder_output = self.encoder(x)
        if self.model_type == "vae":
            mu, logvar = encoder_output
            z = self.reparameterize(mu, logvar)
        else:  # "ae"
            z, _ = encoder_output  # logvar is None

        vq_loss = torch.tensor(0.0, device=x.device)
        indices = None
        if self.use_vq:
            z, vq_loss, indices = self.vq_layer(z)

        recon = self.decoder(z)
        return recon, encoder_output[0], encoder_output[1], z, vq_loss, indices  # mu or z, logvar or None

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim, 1, 1, device=device)
        if self.use_vq:
            z, _, _ = self.vq_layer(z)
        return self.decoder(z)