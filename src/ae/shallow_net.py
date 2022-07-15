import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MixtureSameFamily, Independent, Categorical
from einops import rearrange
from src.utils.util import get_loss

class UNet(nn.Module):
    def __init__(self, z_size=128, in_size=32, loss_fn='mse'):
        super(UNet, self).__init__()
        out_channels = {
            'mse': 3,
            'ce': 256 * 3,
            'normal': 6,
            'gmm': 3 * (2 * 3 + 3)
        }
        self.z_size = z_size
        self.loss_fn = loss_fn
        try:
            self.out_c = out_channels[loss_fn]
        except:
            raise ValueError("No such loss function", loss_fn)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2),
            nn.Flatten()
        )

        # (N, 128 * 4, 1, 1) -> (N, 3, 32, 32)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128 * 4, 64, 3, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, self.out_c, 4, 2),
        )

        self.h = nn.Linear(128 * 4, self.z_size)
        self.dh = nn.Linear(self.z_size, 128 * 4)

    def forward(self, x, sigma=1.0):
        z = self.h(self.encoder(x))
        zh = rearrange(self.dh(z), 'b n -> b n 1 1')
        out = self.decoder(zh)
        y, loss = get_loss(x, out, self.loss_fn, sigma)
        z = z.flatten(1)
        return y, z, loss

    def encode(self, x):
        z = self.h(self.encoder(x))
        return z
