import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MixtureSameFamily, Categorical, Independent
from einops import rearrange


class UNet(nn.Module):
    def __init__(self, z_size=128, in_size=32, blk='basic', loss_fn='mse', kl_tolerance=0.5):
        super(UNet, self).__init__()
        features = 8

        self.loss_fn = loss_fn
        self.z_size = z_size
        self.kl_tolerance = kl_tolerance
        out_channels = {
            'mse': 3,
            'ce': 256 * 3,
            'normal': 6,
            'gmm': 3 * (2 * 3 + 3)
        }
        try:
            self.out_c = out_channels[loss_fn]
        except:
            raise ValueError("No such loss function", loss_fn)
        self.encoder1 = UNet._block(3, features, name="enc1")
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = UNet._block(features * 8, features * 4, name="bottleneck")

        self.h = nn.Linear(features * 4 * 4, self.z_size * 2)
        self.dh = nn.Linear(self.z_size, features * 4 * 4)

        self.upconv4 = nn.ConvTranspose2d(features * 4, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=self.out_c, kernel_size=1)

    def forward(self, x, sigma=1.0, vae_sigma=1.0):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        zs = self.bottleneck(self.pool(enc4)).flatten(1)
        zs = self.h(zs)

        mu = zs[:, :self.z_size]
        log_var = zs[:, self.z_size:]
        z = mu + torch.exp(log_var / 2.0) * torch.randn_like(mu) * vae_sigma

        kl_loss = - 0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        min_kl = torch.zeros_like(kl_loss) + self.kl_tolerance
        kl_loss = torch.max(kl_loss, min_kl).mean()

        dz = rearrange(self.dh(z), 'b (c h w) -> b c h w', h=2, w=2)
        dec4 = self.upconv4(dz)
        dec4 = self.decoder4(torch.cat((dec4, enc4), dim=1))
        dec3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat((dec3, enc3), dim=1))
        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))
        out = self.conv(dec1)


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
 

        zs = [enc1, enc2, enc3, enc4, z]
        zx = torch.cat([x_.flatten(1) for x_ in zs], dim=1)
        return y, zx, loss, kl_loss


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


