import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.simclr.config import Config
from src.simclr.resnet import ResNet
from src.utils.util import CIFAR10Pair

from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm

class SimCLR:
    def __init__(self, **kwargs):
        self.cfg = Config(**kwargs)

        if self.cfg.dset == 'cifar10':
            mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
            self.inp_size = 32
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.inp_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            drop_last = self.cfg.task == 'train'
            train_data = CIFAR10Pair(root='data', train=True, transform=train_transform, download=True)
            self.train_loader = DataLoader(train_data, batch_size=self.cfg.batch_size, shuffle=True, num_workers=16,
                                           pin_memory=True, drop_last=drop_last)

            self.memory_data = CIFAR10Pair(root='data', train=True, transform=test_transform, download=True)
            self.memory_loader = DataLoader(self.memory_data, batch_size=self.cfg.batch_size, shuffle=False,
                                            num_workers=16,
                                            pin_memory=True)

            test_data = CIFAR10Pair(root='data', train=False, transform=test_transform, download=True)
            self.test_loader = DataLoader(test_data, batch_size=self.cfg.batch_size, shuffle=False, num_workers=16,
                                          pin_memory=True)
            self.classes = 10

        self.model = ResNet(self.cfg.z_size)
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-6)
        self.prefix = f"{self.cfg.prefix}_{self.cfg.task}_{self.cfg.dset}"
        self.ckpt = f"ckpt/{self.prefix}"
        self.epoch = 0
        self.steps = 0
        self.best_val_loss = np.Inf

    def train_epoch(self):
        loader = self.train_loader
        self.model.train()
        losses = 0
        for idx, (x1, x2, y) in enumerate(tqdm(loader)):
            x1, x2 = x1.cuda(non_blocking=True), x2.cuda(non_blocking=True)
            feature_1, out_1 = self.model(x1)
            feature_2, out_2 = self.model(x2)
            out = torch.cat([out_1, out_2], dim=0)
            # [2*B, 2*B]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.cfg.temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.cfg.batch_size, device=sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            sim_matrix = sim_matrix.masked_select(mask).view(2 * self.cfg.batch_size, -1)

            # compute loss
            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.cfg.temperature)
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.steps += 1
            losses += loss.item()
        N = len(loader) - 1 if len(loader) > 1 else 1
        losses /= N
        return losses

    def knn(self):
        self.model.eval()
        feature_bank = []
        with torch.no_grad():
            # generate feature bank
            for data, _, target in tqdm(self.memory_loader, desc='Feature extracting'):
                feature, out = self.model(data.cuda(non_blocking=True))
                feature_bank.append(feature)

            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            feature_labels = torch.tensor(self.memory_loader.dataset.targets, device=feature_bank.device)


    def train(self, start_epoch=0):
        for epoch in range(start_epoch, self.cfg.epoch):
            self.epoch = epoch
            train_loss = self.train_epoch()
            val_loss = self.knn()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save("best")
            if (epoch + 1) % self.cfg.epoch_ckpt_freq == 0:
                self.save(f"e{epoch:03d}")
            print(f"Epoch {epoch:3d}, Train Loss: {train_loss:6.5f}, Val Loss: {val_loss:6.5f}")

    def save(self, fname):
        pass

    def load(self, ckpt=None):
        pass

    def train_val_epoch_probing(self, linear, optimizer, train=True):
        pass

    def linear_probing(self):
        pass

    def tsne(self):
        pass

    def run(self):
        if self.cfg.task == 'train':
            self.writer = SummaryWriter(f'logs/{self.prefix}')
            self.train()
        elif self.cfg.task == 'probing':
            self.writer = SummaryWriter(f'logs/{self.prefix}')
            self.load()
            self.linear_probing()
        elif self.cfg.task == 'tsne':
            self.load()
            self.tsne()
        else:
            raise ValueError("No such task", self.cfg.task)

