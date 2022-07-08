import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal
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