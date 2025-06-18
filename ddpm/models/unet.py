import torch
import torch.nn as nn
import math
from models.blocks import ResnetBlock, AttnBlock, Downsample, Upsample, swish

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  # (B, embedding_dim)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder (Downsampling path)
        self.downs = nn.ModuleList([
            # Level 1: [64, 64] -> [32, 32], 64 channels
            nn.ModuleList([
                ResnetBlock(base_channels, base_channels, time_emb_dim=time_emb_dim),
                ResnetBlock(base_channels, base_channels, time_emb_dim=time_emb_dim),
                AttnBlock(base_channels, base_channels),
                Downsample(base_channels)
            ]),
            # Level 2: [32, 32] -> [16, 16], 128 channels
            nn.ModuleList([
                ResnetBlock(base_channels, base_channels * 2, time_emb_dim=time_emb_dim),
                ResnetBlock(base_channels * 2, base_channels * 2, time_emb_dim=time_emb_dim),
                AttnBlock(base_channels * 2, base_channels * 2),
                Downsample(base_channels * 2)
            ]),
            # Level 3: [16, 16] -> [8, 8], 256 channels
            nn.ModuleList([
                ResnetBlock(base_channels * 2, base_channels * 4, time_emb_dim=time_emb_dim),
                ResnetBlock(base_channels * 4, base_channels * 4, time_emb_dim=time_emb_dim),
                AttnBlock(base_channels * 4, base_channels * 4),
                Downsample(base_channels * 4)
            ])
        ])

        # Bottleneck: [8, 8], 512 channels
        self.bottleneck = nn.ModuleList([
            ResnetBlock(base_channels * 4, base_channels * 8, time_emb_dim=time_emb_dim),
            AttnBlock(base_channels * 8, base_channels * 8),
            ResnetBlock(base_channels * 8, base_channels * 8, time_emb_dim=time_emb_dim)
        ])

        # Decoder (Upsampling path)
        self.ups = nn.ModuleList([
            # Level 3: [8, 8] -> [16, 16], 256 channels (after concat with 256 from down)
            nn.ModuleList([
                Upsample(base_channels * 8),
                ResnetBlock(base_channels * 12, base_channels * 4, time_emb_dim=time_emb_dim),  # 512+256=768 -> 256
                ResnetBlock(base_channels * 4, base_channels * 4, time_emb_dim=time_emb_dim),
                AttnBlock(base_channels * 4, base_channels * 4)
            ]),
            # Level 2: [16, 16] -> [32, 32], 128 channels (after concat with 128 from down)
            nn.ModuleList([
                Upsample(base_channels * 4),
                ResnetBlock(base_channels * 6, base_channels * 2, time_emb_dim=time_emb_dim),  # 256+128=384 -> 128
                ResnetBlock(base_channels * 2, base_channels * 2, time_emb_dim=time_emb_dim),
                AttnBlock(base_channels * 2, base_channels * 2)
            ]),
            # Level 1: [32, 32] -> [64, 64], 64 channels (after concat with 64 from down)
            nn.ModuleList([
                Upsample(base_channels * 2),
                ResnetBlock(base_channels * 3, base_channels, time_emb_dim=time_emb_dim),  # 128+64=192 -> 64
                ResnetBlock(base_channels, base_channels, time_emb_dim=time_emb_dim),
                AttnBlock(base_channels, base_channels)
            ])
        ])

        # Final convolution to output channels
        self.final_conv = nn.ModuleList([
            ResnetBlock(base_channels, base_channels, time_emb_dim=time_emb_dim),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        ])

    def forward(self, x, t):
        expected_output_shape = x.shape  # Expected: [batch_size, 3, 64, 64]
        t_emb = self.time_mlp(t)  # (B, time_emb_dim)
        h = self.init_conv(x)

        # Encoder
        hs = []
        for i, (res1, res2, attn, down) in enumerate(self.downs):
            h = res1(h, t_emb)
            h = res2(h, t_emb)
            h = attn(h)
            hs.append(h)
            h = down(h)

        # Bottleneck
        h = self.bottleneck[0](h, t_emb)
        h = self.bottleneck[1](h)
        h = self.bottleneck[2](h, t_emb)

        # Decoder
        for i, (up, res1, res2, attn) in enumerate(self.ups):
            h = up(h)
            h_skip = hs.pop()
            # Assert skip connection matches upsampled size
            assert h.shape[2:] == h_skip.shape[2:], \
                f"Skip connection size mismatch at Up {i+1}: expected {h.shape[2:]}, got {h_skip.shape[2:]}"
            h = torch.cat([h, h_skip], dim=1)
            h = res1(h, t_emb)
            h = res2(h, t_emb)
            h = attn(h)

        # Final convolution
        h = self.final_conv[0](h, t_emb)
        h = self.final_conv[1](h)

        # Assert output shape matches input shape
        if h.shape != expected_output_shape:
            raise ValueError(
                f"UNet output shape mismatch: expected {expected_output_shape}, got {h.shape}"
            )

        return h

# Test
if __name__ == "__main__":
    model = UNet()
    x = torch.randn(4, 3, 64, 64)  # B, C, H, W
    t = torch.randint(0, 1000, (4,))
    out = model(x, t)