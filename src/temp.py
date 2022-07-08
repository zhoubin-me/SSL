import torch
from src.ae.resnet import ResVAE

if __name__ == '__main__':
    model = ResVAE()
    x = torch.rand(128, 3, 32, 32)
    y, z, loss = model(x)
    print(y.size())