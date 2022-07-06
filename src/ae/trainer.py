import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.datasets import CelebA, CIFAR10
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from src.ae.config import Config
# from src.ae.network import Network
from src.ae.unet import UNet
from src.ae.unet_no_pyr import UNet as UNetNoPyr

import os
import argparse
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, **kwargs):
        self.cfg = Config(**kwargs)
        if self.cfg.dset == 'cifar10':
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ])
            self.dset_train = CIFAR10(self.cfg.dset_root, train=True, download=True, transform=train_transform)
            self.dset_val = CIFAR10(self.cfg.dset_root, train=False, download=True, transform=transforms.ToTensor())
            self.in_size = 32
            self.classes = 10

        elif self.cfg.dset == 'celeba':
            train_transform = transforms.Compose([
                transforms.Resize(178),
                transforms.CenterCrop(178),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ])

            val_transform = transforms.Compose([
                transforms.Resize(178),
                transforms.CenterCrop(178),
                transforms.ToTensor(),
            ])

            self.dset_train = CelebA(self.cfg.dset_root, split='train', target_type='attr', transform=train_transform)
            self.dset_val = CelebA(self.cfg.dset_root, split='val', target_type='attr', transform=val_transform)
            self.in_size = 178
            self.classes = 40
        else:
            raise NotImplementedError("No such dataset:", self.cfg.dset)

        self.train_loader = DataLoader(self.dset_train, self.cfg.batch_size,
                                       shuffle=True, num_workers=9, pin_memory=True)
        self.val_loader = DataLoader(self.dset_val, self.cfg.batch_size,
                                     shuffle=False, num_workers=9, pin_memory=True)

        if self.cfg.model == 'unet':
            self.model = UNet(self.cfg.z_size, self.in_size, self.cfg.blk)
        elif self.cfg.model == 'unet_no_pyr':
            self.model = UNetNoPyr(self.cfg.z_size, self.in_size, self.cfg.blk)
        else:
            raise ValueError("No such model", self.cfg.model)

        self.model = nn.DataParallel(self.model).cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
        self.prefix = f"{self.cfg.prefix}_{self.cfg.task}_{self.cfg.dset}_{self.cfg.model}"
        self.ckpt = f"ckpt/{self.prefix}"
        self.epoch = 0
        self.steps = 0

    def train_val_epoch(self, train=False):
        loader = self.train_loader if train else self.val_loader
        prefix = 'train' if train else 'val'
        self.model.train() if train else self.model.eval()
        losses = 0
        for i, (x, _) in enumerate(tqdm(loader)):
            x = x.cuda()
            if train:
                _, _, log_prob = self.model(x)
                loss = log_prob.neg().mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar(f"loss/train_batch_end", loss.item(), self.steps)
                self.steps += 1
            else:
                with torch.no_grad():
                    _, _, log_prob = self.model(x)
                    loss = log_prob.neg().mean()
            losses += loss.item()
        losses /= len(loader)
        self.writer.add_scalar(f'loss/{prefix}_epoch_end', losses, self.epoch)
        return losses

    def train(self, start_epoch=0):
        best_val_loss = np.Inf
        for epoch in range(start_epoch, self.cfg.epoch):
            self.epoch = epoch
            train_loss = self.train_val_epoch(True)
            val_loss = self.train_val_epoch(False)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save("best")
            if (epoch + 1) % self.cfg.epoch_ckpt_freq == 0:
                self.save(f"e{epoch:03d}")
                self.reconstruct()
            print(f"Epoch {epoch:3d}, Train Loss: {train_loss:6.5f}, Val Loss: {val_loss:6.5f}")

    def save(self, fname):
        if not os.path.exists(self.ckpt):
            os.mkdir(self.ckpt)
        torch.save(
            {
                "epoch": self.epoch,
                "model": self.model.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }, f"{self.ckpt}/{fname}.pth.tar"
        )

    def load(self):
        if os.path.exists(self.cfg.load_ckpt):
            data = torch.load(self.cfg.load_ckpt)
            self.model.module.load_state_dict(data['model'])
            self.optimizer.load_state_dict(data['optimizer'])
            self.epoch = data['epoch']
        else:
            raise ValueError("No such path", self.cfg.load_ckpt)

    def train_val_epoch_probing(self, linear, optimizer, train=True):
        loader = self.train_loader if train else self.val_loader
        losses, corrects, total = 0, 0, 0
        self.model.eval()
        for i, (x, y) in enumerate(loader):
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                z = self.model.module.encode(x.cuda())

            logits = linear(z)
            loss = F.cross_entropy(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            corrects += torch.sum((logits.argmax(dim=-1) == y).float()).item()
            losses += loss.item()
            total += x.size(0)
        loss = losses / len(loader)
        accuracy = corrects / total
        return loss, accuracy

    def linear_probing(self):
        x = torch.rand(1, 3, 32, 32).cuda()
        y = self.model.module.encode(x)
        N = y.size(1)
        linear = nn.Linear(N, self.classes).cuda()
        optimizer = torch.optim.AdamW(linear.parameters())
        self.load()
        for epoch in range(self.cfg.epoch_probing):
            train_loss, train_acc = self.train_val_epoch_probing(linear, optimizer, train=True)
            val_loss, val_acc = self.train_val_epoch_probing(linear, optimizer, train=False)
            self.writer.add_scalar(f'loss/train_epoch_end', train_loss, epoch)
            self.writer.add_scalar(f'accuracy/train_epoch_end', train_acc, epoch)
            self.writer.add_scalar(f'loss/val_epoch_end', val_loss, epoch)
            self.writer.add_scalar(f'accuracy/val_epoch_end', val_acc, epoch)
            print(f"Probing Epoch {epoch:3d}, Train Acc:{train_acc:.4f} Loss:{train_loss:.4f}\t"
                  f"Val Acc: {val_acc:.4f} Loss {val_loss:.4f}")

    def reconstruct(self):
        path = os.path.join("imgs", self.prefix)
        if not os.path.exists(path):
            os.mkdir(path)
        for i, (x, _) in enumerate(self.val_loader):
            with torch.no_grad():
                x = x.cuda()
                y, _, _ = self.model(x, sigma=0.01)
            z = torch.cat((x, y), dim=2)
            img = make_grid(z[:25], 5)
            img = transforms.functional.to_pil_image(img)
            img.save(os.path.join(path, f"img_{i:03d}.png"))
            if i > 20:
                break

    def tsne(self):
        data = []
        for idx, (x, y) in enumerate(self.val_loader):
            z = self.model.module.encode(x.cuda())
            data.append((z.cpu(), y))
            if idx > 10:
                break

        xx = torch.cat(list(x[0] for x in data), dim=0)
        yy = torch.cat(list(x[1] for x in data), dim=1)
        xx = xx.view(xx.size(0), -1)

        tsne = TSNE(n_components=2, verbose=True, perplexity=40, n_iter=1000, learning_rate='auto')
        xxx = tsne.fit_transform(xx)

        plt.figure(figsize=(12, 8))
        plt.scatter(xxx[:, 0], xxx[:, 1], c=yy, cmap=plt.cm.get_cmap('Set1'))
        path = os.path.join("imgs", self.prefix)
        if not os.path.exists(path):
            os.mkdir(path)
        plt.savefig(os.path.join(path, 'tsne.png'))

    def run(self):
        if self.cfg.task == 'train':
            self.prefix += "_train"
            self.writer = SummaryWriter(f'logs/{self.prefix}')
            self.train()
        elif self.cfg.task == 'probing':
            self.prefix += "_probing"
            self.writer = SummaryWriter(f'logs/{self.prefix}')
            self.linear_probing()
        elif self.cfg.task == 'tsne':
            self.load()
            self.tsne()
        elif self.cfg.task == 'decode':
            self.load()
            self.reconstruct()
        else:
            raise ValueError("No such task", self.cfg.task)


def main():
    cfg = Config()
    parser = argparse.ArgumentParser()
    for k, v in cfg.__dict__.items():
        if type(v) == bool:
            parser.add_argument(f'--{k}', default=True, type=lambda x: (str(x).lower() == 'true'))
        else:
            parser.add_argument(f'--{k}', type=type(v), default=v)
    args = parser.parse_args()
    kwargs = args.__dict__
    print(kwargs)
    trainer = Trainer(**kwargs)
    trainer.run()
