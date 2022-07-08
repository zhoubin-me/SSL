from torchvision.models.resnet import resnet50, resnet18, resnet34

import torch
import torch.nn as nn

class ResVAE(nn.Module):
    def __init__(self, z_size=128, in_size=32, blk='basic', loss_fn='mse', kl_tolerance=0.5):
        super(ResVAE, self).__init__()
        self.z_size = z_size

        model = resnet18()
        self.f = []
        for name, module in model.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.Linear) or isinstance(module, nn.MaxPool2d) or \
                    isinstance(module, nn.AdaptiveAvgPool2d):
                continue
            self.f.append(module)
        self.f.append(nn.Flatten())
        self.f = nn.Sequential(*self.f)
        self.h = nn.Linear(8192, self.z_size)



    def forward(self, x):
        return self.f(x)

if __name__ == '__main__':
    model = ResVAE()
    x = torch.rand(128, 3, 32, 32)
    y = model(x)
    print(y.size())