import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, z_size=128, loss_fn='mse'):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(16 * 16 * 16 * 64, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, z_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_size, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 16 * 16 * 16 * 64)
        )

    def forward(self, x, sigma=1.0):
        z = self.encoder(x)
        out = self.decoder(z)
        loss = F.mse_loss(x, out)
        return out, z, loss


