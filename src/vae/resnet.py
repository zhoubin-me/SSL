import torch
import torch.nn as nn

from torchvision.models.resnet import resnet50, resnet18, resnet34

from einops import rearrange

from src.utils.util import get_loss


class ResNetVAE(nn.Module):
    def __init__(self, z_size=128, in_size=32, loss_fn='mse', kl_tolerance=0.5):
        super(ResNetVAE, self).__init__()
        self.z_size = z_size
        out_channels = {
            'mse': 3,
            'ce': 256 * 3,
            'normal': 6,
            'gmm': 3 * (2 * 3 + 3)
        }
        self.kl_tolerance = kl_tolerance
        self.loss_fn = loss_fn
        try:
            self.out_c = out_channels[loss_fn]
        except:
            raise ValueError("No such loss function", loss_fn)

        model = resnet18()
        self.encoder = []
        for name, module in model.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.Linear) or isinstance(module, nn.MaxPool2d):
                continue
            self.encoder.append(module)
        self.encoder.append(nn.Flatten())
        self.encoder = nn.Sequential(*self.encoder)
        self.h = nn.Linear(512, self.z_size * 2)

        self.dh = nn.Linear(self.z_size, 512 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, self.out_c, 2, 2),
        )

    def forward(self, x, sigma=1.0, vae_sigma=1.0):
        zs = self.h(self.encoder(x))
        mu = zs[:, :self.z_size]
        log_var = zs[:, self.z_size:]
        z = mu + torch.exp(log_var / 2.0) * torch.randn_like(mu) * vae_sigma

        kl_loss = - 0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        min_kl = torch.zeros_like(kl_loss) + self.kl_tolerance
        kl_loss = torch.max(kl_loss, min_kl).mean()
        dz = rearrange(self.dh(z), 'b (n h w) -> b n h w', h=2, w=2)
        out = self.decoder(dz)
        y, loss = get_loss(x, out, self.loss_fn, sigma)
        return y, z, loss, kl_loss