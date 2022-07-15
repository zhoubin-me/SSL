import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from torchvision.models import resnet
from src.simclr.config import Config
from src.simclr.resnet import ResNet
from src.utils.util import CIFAR10Pair

import numpy as np
from tqdm import tqdm
import os


class Trainer:
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
            self.train_data = CIFAR10Pair(root=self.cfg.dset_root, train=True, transform=train_transform, download=True)
            self.train_loader = DataLoader(self.train_data, batch_size=self.cfg.batch_size, shuffle=True, num_workers=16,
                                           pin_memory=True, drop_last=drop_last)

            self.memory_data = CIFAR10Pair(root=self.cfg.dset_root, train=True, transform=test_transform, download=True)
            self.memory_loader = DataLoader(self.memory_data, batch_size=self.cfg.batch_size, shuffle=False,
                                            num_workers=16,
                                            pin_memory=True)

            self.test_data = CIFAR10Pair(root=self.cfg.dset_root, train=False, transform=test_transform, download=True)
            self.test_loader = DataLoader(self.test_data, batch_size=self.cfg.batch_size, shuffle=False, num_workers=16,
                                          pin_memory=True)
            self.classes = 10

        try:
            net = getattr(resnet, self.cfg.model)()
        except:
            raise ValueError("No such network", self.cfg.model)

        self.model = ResNet(net, self.cfg.z_size)
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.prefix = f"{self.cfg.prefix}_{self.cfg.task}_{self.cfg.dset}"
        self.ckpt = f"ckpt/{self.prefix}"
        self.epoch = 0
        self.steps = 0
        self.k = 200
        self.best_val_acc = np.NINF

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
            self.writer.add_scalar("loss/train_batch_end", loss.item(), self.epoch * len(loader) + idx)

        N = len(loader) - 1 if len(loader) > 1 else 1
        losses /= N
        return losses

    def knn(self):
        self.model.eval()
        corrects, total_num = 0, 0
        k = self.k
        feature_bank = []
        with torch.no_grad():
            # generate feature bank
            for data, _, target in tqdm(self.memory_loader, desc='Feature extracting'):
                feature, out = self.model(data.cuda(non_blocking=True))
                feature_bank.append(feature)

            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            feature_labels = torch.tensor(self.memory_data.targets, device=feature_bank.device)
            for data, _, target in tqdm(self.test_loader):
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                feature, out = self.model(data)

                total_num += data.size(0)
                # compute cos similarity between each feature vector and feature bank ---> [B, N]
                sim_matrix = torch.mm(feature, feature_bank)
                # [B, K]
                sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
                # [B, K]
                sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
                sim_weight = (sim_weight / self.cfg.temperature).exp()

                # counts for each class
                one_hot_label = torch.zeros(data.size(0) * k, 10, device=sim_labels.device)
                # [B*K, C]
                one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
                # weighted score ---> [B, C]
                pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, self.classes) * sim_weight.unsqueeze(dim=-1), dim=1)

                pred_labels = pred_scores.argsort(dim=-1, descending=True)
                corrects += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            accuracy = corrects / total_num

        return accuracy

    def train(self, start_epoch=0):
        for epoch in range(start_epoch, self.cfg.epoch):
            self.epoch = epoch
            train_loss = self.train_epoch()
            val_acc = self.knn()
            self.writer.add_scalar('loss/train_epoch_end', train_loss, epoch)
            self.writer.add_scalar('accuracy/knn_val', val_acc, epoch)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save("best")
            if (epoch + 1) % self.cfg.epoch_ckpt_freq == 0:
                self.save(f"e{epoch:03d}")
            print(f"Epoch {epoch:3d}, Train Loss: {train_loss:6.5f}, Val Acc: {val_acc:6.5f}")

    def save(self, fname):
        if not os.path.exists(self.ckpt):
            os.mkdir(self.ckpt)
        torch.save({
            'best_val_acc': self.best_val_acc,
            'epoch': self.epoch,
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, f"{self.ckpt}/{fname}.pth.tar")

    def load(self, ckpt=None):
        if ckpt is None:
            ckpt = self.cfg.load_ckpt
        if os.path.exists(ckpt):
            data = torch.load(ckpt)
            self.model.module.load_state_dict(data['model'])
            self.optimizer.load_state_dict(data['optimizer'])
        else:
            raise ValueError("No such path", self.cfg.load_ckpt)

    def train_val_epoch_probing(self, linear, optimizer, train=True):
        loader = self.train_loader if train else self.test_loader
        losses, corrects, total = 0, 0, 0
        self.model.eval()
        for i, (x, _, y) in enumerate(loader):
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                z = self.model.module.f(x)
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
        linear = nn.Linear(self.cfg.z_size, self.classes).cuda()
        optimizer = torch.optim.AdamW(linear.parameters())
        for epoch in range(self.cfg.epoch_probing):
            train_loss, train_acc = self.train_val_epoch_probing(linear, optimizer, train=True)
            val_loss, val_acc = self.train_val_epoch_probing(linear, optimizer, train=False)
            self.writer.add_scalar(f'loss/train_epoch_end', train_loss, epoch)
            self.writer.add_scalar(f'accuracy/train_epoch_end', train_acc, epoch)
            self.writer.add_scalar(f'loss/val_epoch_end', val_loss, epoch)
            self.writer.add_scalar(f'accuracy/val_epoch_end', val_acc, epoch)
            print(f"Probing Epoch {epoch:3d}, Train Acc:{train_acc:.4f} Loss:{train_loss:.4f}\t"
                  f"Val Acc: {val_acc:.4f} Loss {val_loss:.4f}")

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

