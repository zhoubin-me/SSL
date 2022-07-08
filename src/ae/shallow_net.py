import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MixtureSameFamily, Independent, Categorical
from einops import rearrange

class UNet(nn.Module):
    def __init__(self, z_size=128, in_size=32, blk='basic', loss_fn='mse'):
        super(UNet, self).__init__()
        features = 8
        out_channels = {
            'mse': 3,
            'ce': 256 * 3,
            'normal': 6,
            'gmm': 3 * (2 * 3 + 3)
        }
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

        self.h = nn.Linear(128 * 4, 128)
        self.dh = nn.Linear(128, 128 * 4)

    def forward(self, x, sigma=1.0):
        z = self.h(self.encoder(x))
        zh = rearrange(self.dh(z), 'b n -> b n 1 1')
        out = self.decoder(zh)

        if self.loss_fn == 'mse':
            y = F.sigmoid(out)
            loss = F.mse_loss(x, y)
        elif self.loss_fn == 'ce':
            loss = F.cross_entropy(rearrange(out, 'b (c n) h w -> (b c h w) n', c=3), (x.view(-1) * 255.0).long())
            y = rearrange(out, 'b (c n) h w -> b c n h w', c=3)
            y = y.argmax(dim=2).float().div(255.0)
        elif self.loss_fn == 'normal':
            mu = out[:, :3, :, :]
            log_var = out[:, 3:, :, :]
            dist = Normal(mu, log_var.exp() * sigma)
            loss = dist.log_prob(2 * x - 1).mean().neg()
            y = dist.sample()
            y = torch.clamp(y, -1, 1)
            y = y / 2 + 0.5
        elif self.loss_fn == 'gmm':
            out = rearrange(out, 'b (c n) h w -> (b c h w) n', c=3)
            mix = Categorical(logits=out[:, :3])
            comp = Independent(Normal(out[:, 3:6].unsqueeze(-1), out[:, 6:9].unsqueeze(-1).exp() * sigma), 1)
            gmm = MixtureSameFamily(mix, comp)
            loss = gmm.log_prob(2 * x.view(-1).unsqueeze(-1) - 1).mean().neg()
            y = gmm.sample()
            y = rearrange(y, '(b c h w) 1 -> b c h w', c=3, h=32, w=32)
            y = torch.clamp(y, -1, 1)
            y = y / 2 + 0.5
        else:
            raise ValueError("No such loss function", self.loss_fn)

        z = z.flatten(1)
        return y, z, loss

    def encode(self, x):
        z = self.h(self.encoder(x))
        return z

    @staticmethod
    def _block(in_channels, features, name=None):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )


