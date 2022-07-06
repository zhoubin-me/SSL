import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from einops import rearrange

class UNet(nn.Module):
    def __init__(self, z_size=128, in_size=32, blk='basic'):
        super(UNet, self).__init__()
        features = 3
        self.encoder1 = UNet._block(3, features, name="enc1")
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=256 * 3, kernel_size=1)

    def forward(self, x, sigma=1.0):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        z = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(z)
        dec4 = self.decoder4(torch.cat((dec4, enc4), dim=1))
        dec3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat((dec3, enc3), dim=1))
        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))
        out = self.conv(dec1)

        log_prob = F.cross_entropy(rearrange(out, 'b (c n) h w -> (b c h w) n', c=3), (x.view(-1) * 255.0).long()).neg()
        y = rearrange(out, 'b (c n) h w -> b c n h w', c=3)
        y = y.argmax(dim=2).float().div(255.0)

        B = x.size(0)
        z = torch.cat((enc1.view(B, -1), enc2.view(B, -1), enc3.view(B, -1), enc4.view(B, -1), z.view(B, -1)), dim=1)
        return y, z, log_prob

    def encode(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        z = self.bottleneck(self.pool(enc4))

        B = x.size(0)
        out = torch.cat((enc1.view(B, -1), enc2.view(B, -1), enc3.view(B, -1), enc4.view(B, -1), z.view(B, -1)), dim=1)
        return out

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


