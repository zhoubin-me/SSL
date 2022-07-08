
from torchvision.datasets import CIFAR10
import numpy as np
from PIL import Image

class CIFAR10Few(CIFAR10):
    def few(self, percentage=0.1):
        self.percentage = percentage
        self.new_idxs = []
        for i, _ in enumerate(self.classes):
            indexs = [j for j, x in enumerate(self.targets) if x == i]
            n = int(len(indexs) * percentage)
            self.new_idxs.extend(np.random.choice(indexs, size=n))

    def __len__(self):
        return len(self.new_idxs)

    def __getitem__(self, idx):
        new_idx = self.new_idxs[idx]
        img, target = self.data[new_idx], self.targets[new_idx]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

if __name__ == '__main__':
    dset = CIFAR10Few('dataset', download=True, train=True)
    dset.few()
