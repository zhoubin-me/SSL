from dataclasses import dataclass

@dataclass
class Config:
    lr: float = 0.001
    model: str = 'resnet50'
    z_size: int = 128
    dset: str = 'cifar10'
    dset_root: str = 'dataset'
    batch_size: int = 1024
    epoch: int = 1000
    epoch_probing: int = 100
    epoch_ckpt_freq: int = 10
    prefix: str = "simclr"
    load_ckpt: str = "simclr"
    task: str = "train"
    temperature: float = 0.5
