from dataclasses import dataclass

@dataclass
class Config:
    lr: float = 0.001
    z_size: int = 128
    few_perc: float = 1.0
    blk: str = 'basic'
    model: str = 'unet'
    dset: str = 'cifar10'
    dset_root: str = 'dataset'
    batch_size: int = 128
    epoch: int = 200
    epoch_probing: int = 100
    epoch_ckpt_freq: int = 10
    prefix: str = "ae"
    load_ckpt: str = "ae"
    task: str = "train"
    loss_fn: str = 'normal'

