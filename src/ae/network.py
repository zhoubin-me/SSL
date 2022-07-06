import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchsummary import summary
from torch.distributions import Normal


class BasicBlock(nn.Module):
    def __init__(self, in_channels, features, out_channels, stride=1):
        super().__init__()
        if stride >= 1:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        elif stride <= -1:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 2, -stride, 0, bias=False)
        else:
            raise NotImplementedError("No such stride:", stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, features, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, features, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(features)

        if stride >= 1:
            self.conv2 = nn.Conv2d(features, features, 3, stride, 1, bias=False)
        elif stride <= -1:
            self.conv2 = nn.ConvTranspose2d(features, features, 2, -stride, 0, bias=False)
        else:
            raise NotImplementedError("no such stride:", stride)

        self.bn2 = nn.BatchNorm2d(features)
        self.conv3 = nn.Conv2d(features, out_channels, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        return out


class Network(nn.Module):
    def __init__(self, z_size, in_size=32, blk='basic'):
        super().__init__()

        self.z_size = z_size
        if blk == 'basic':
            block = BasicBlock
        elif blk == 'bottlent':
            block = Bottleneck
        else:
            raise NotImplementedError("No such block:", blk)

        self.enc1 = block(3, 8, 16, 2)
        self.enc2 = block(16, 8, 32, 2)
        self.enc3 = block(32, 16, 64, 2)
        self.enc4 = block(64, 32, 128, 2)

        self.dec1 = block(128, 32, 64, -2)
        self.dec2 = block(64, 16, 32, -2)
        self.dec3 = block(32, 8, 16, -2)
        self.dec4 = block(16, 8, 3 * 2, -2)

        self.h_size = int(in_size / (2 ** 4))
        self.fc1 = nn.Linear(self.h_size * self.h_size * 128, self.z_size)
        self.fc2 = nn.Linear(self.z_size, self.h_size * self.h_size * 128)

    def forward(self, x, sigma=1.0):
        z = self.encode(x)
        y, dist = self.decode(z, sigma)
        log_prob = dist.log_prob(2 * x - 1)
        return y, z, log_prob

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.fc1(x)
        return x

    def decode(self, h, sigma=1.0):
        h = self.fc2(h)
        h = rearrange(h, 'b (c h w) -> b c h w', c=128, h=self.h_size, w=self.h_size)
        y = self.dec1(h)
        y = self.dec2(y)
        y = self.dec3(y)
        y = self.dec4(y)
        mu = y[:, :3, :, :]
        log_var = y[:, 3:, :, :]
        dist = Normal(mu, log_var.exp() * sigma)
        out = dist.sample()
        out = torch.clamp(out, -1, 1.)
        out = out / 2. + 0.5
        return out, dist


if __name__ == '__main__':
    model = Network(128)
    x = torch.rand(16, 3, 32, 32)
    y, z, dist = model(x)
    print(y.shape)

