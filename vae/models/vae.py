# models/vae.py
import torch
import torch.nn as nn

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        # z: [B, C, H, W] -> [B, H, W, C]
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)
        
        # Distances to codebook
        distances = (z_flattened ** 2).sum(dim=1, keepdim=True) + \
                    (self.embedding.weight ** 2).sum(dim=1) - \
                    2 * torch.matmul(z_flattened, self.embedding.weight.t())
        
        # Quantize
        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)
        
        # Loss terms
        commitment_loss = self.beta * ((z.detach() - z_q) ** 2).mean()
        embedding_loss = ((z - z_q.detach()) ** 2).mean()
        vq_loss = commitment_loss + embedding_loss
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        return z_q.permute(0, 3, 1, 2), vq_loss, encoding_indices

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, constant_variance, input_size):
        super().__init__()
        self.constant_variance = constant_variance
        num_layers = len(hidden_dims)
        min_size = input_size // (2 ** num_layers)
        if min_size < 1:
            raise ValueError(
                f"Input size {input_size}x{input_size} with {num_layers} layers "
                f"reduces spatial size to {min_size}x{min_size}."
            )
        layers = []
        in_channels = input_dim
        for out_channels in hidden_dims:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, padding_mode="reflect"),
                nn.GroupNorm(16, out_channels),
                nn.GELU()
            ])
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Conv2d(hidden_dims[-1], latent_dim, 1)
        if not constant_variance:
            self.fc_logvar = nn.Conv2d(hidden_dims[-1], latent_dim, 1)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        if self.constant_variance:
            logvar = torch.full_like(mu, torch.log(torch.tensor(0.1)))
        else:
            logvar = 5.0 * torch.tanh(self.fc_logvar(x))
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, input_size):
        super().__init__()
        num_layers = len(hidden_dims)
        required_size = 2 ** num_layers
        if input_size % required_size != 0:
            raise ValueError(
                f"Input size {input_size}x{input_size} must be divisible by {required_size}x{required_size} "
                f"for {num_layers} layers."
            )
        self.fc = nn.Conv2d(latent_dim, hidden_dims[-1], 1)
        layers = []
        for i in range(len(hidden_dims) - 1, 0, -1):
            layers.extend([
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i-1], 4, 2, 1),
                nn.GroupNorm(16, hidden_dims[i-1]),
                nn.GELU()
            ])
        layers.append(nn.ConvTranspose2d(hidden_dims[0], output_dim, 4, 2, 1))
        layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        z = self.fc(z)
        return self.decoder(z)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, constant_variance=False, input_size=256, use_vq=False,
                 vq_num_embeddings=512, vq_embedding_dim=64, vq_beta=0.25):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim if not use_vq else vq_embedding_dim
        self.use_vq = use_vq
        
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        
        self.encoder = Encoder(input_dim, hidden_dims, self.latent_dim, constant_variance, input_size)
        self.decoder = Decoder(self.latent_dim, hidden_dims, input_dim, input_size)
        self.vq_layer = VectorQuantizer(vq_num_embeddings, vq_embedding_dim, vq_beta) if use_vq else None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        if x.shape[2] != self.input_size or x.shape[3] != self.input_size:
            raise ValueError(
                f"Input tensor size {x.shape[2]}x{x.shape[3]} does not match "
                f"expected input_size {self.input_size}x{self.input_size}"
            )
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        
        vq_loss = torch.tensor(0.0, device=x.device)
        if self.use_vq:
            z, vq_loss, _ = self.vq_layer(z)
        
        recon = self.decoder(z)
        return recon, mu, logvar, z, vq_loss