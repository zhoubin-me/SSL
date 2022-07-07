import os
from itertools import product
from src.ae.trainer import Trainer
from src.ae.config import Config

loss_fns = ['mse', 'ce', 'normal']
models = ['unet']

for loss_fn, model in product(loss_fns, models):
    cfg = Config()
    cfg.model = model
    cfg.loss_fn = loss_fn
    cfg.load_ckpt = f"ckpt/ae_{loss_fn}_train_cifar10_{model}/best.pth.tar"

    cfg.task = "train"
    trainer = Trainer(**cfg.__dict__)
    trainer.run()

    cfg.task = "decode"
    trainer = Trainer(**cfg.__dict__)
    trainer.run()

    cfg.task = "probing"
    trainer = Trainer(**cfg.__dict__)
    trainer.run()

    cfg.task = "tsne"
    trainer = Trainer(**cfg.__dict__)
    trainer.run()