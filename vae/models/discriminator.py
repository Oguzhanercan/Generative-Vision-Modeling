import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        in_channels = input_dim
        for out_channels in hidden_dims:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 4, 2, 1),
                nn.GroupNorm(16, out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_channels = out_channels
        layers.append(nn.Conv2d(hidden_dims[-1], 1, 4, 1, 0))  # Output 1x1 score
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)