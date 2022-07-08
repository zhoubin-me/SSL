
from torchvision.datasets import CIFAR10
import numpy as np
from PIL import Image



if __name__ == '__main__':
    dset = CIFAR10Few('dataset', download=True, train=True)
    dset.few()
