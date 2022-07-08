import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18

class ResNet(nn.Module):
    def __init__(self, z_size=128):
        super(ResNet, self).__init__()
        self.z_size = z_size
        self.f = []
        model = resnet18()
        for name, module in model.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        self.f.extend([nn.Flatten(), nn.Linear(512 * 4, self.z_size)])
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(self.z_size, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, self.z_size, bias=True))

    def forward(self, x):
        feature = self.f(x)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)