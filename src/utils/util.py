import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

from torchvision.datasets import CIFAR10

from PIL import Image
from einops import rearrange

def get_loss(x, out, loss_fn, sigma):
    if loss_fn == 'mse':
        y = F.sigmoid(out)
        loss = F.mse_loss(x, y)
    elif loss_fn == 'ce':
        loss = F.cross_entropy(rearrange(out, 'b (c n) h w -> (b c h w) n', c=3), (x.view(-1) * 255.0).long())
        y = rearrange(out, 'b (c n) h w -> b c n h w', c=3)
        y = y.argmax(dim=2).float().div(255.0)
    elif loss_fn == 'normal':
        mu = out[:, :3, :, :]
        log_var = out[:, 3:, :, :]
        dist = Normal(mu, log_var.exp() * sigma)
        loss = dist.log_prob(2 * x - 1).mean().neg()
        y = dist.sample()
        y = torch.clamp(y, -1, 1)
        y = y / 2 + 0.5
    elif loss_fn == 'gmm':
        out = rearrange(out, 'b (c n) h w -> (b c h w) n', c=3)
        mix = Categorical(logits=out[:, :3])
        comp = Independent(Normal(out[:, 3:6].unsqueeze(-1), out[:, 6:9].unsqueeze(-1).exp() * sigma), 1)
        gmm = MixtureSameFamily(mix, comp)
        loss = gmm.log_prob(2 * x.view(-1).unsqueeze(-1) - 1).mean().neg()
        y = gmm.sample()
        y = rearrange(y, '(b c h w) 1 -> b c h w', c=3, h=32, w=32)
        y = torch.clamp(y, -1, 1)
        y = y / 2 + 0.5
    else:
        raise ValueError("No such loss function", loss_fn)

    return y, loss

class CIFAR10Pair(CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

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