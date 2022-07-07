import os
from itertools import product

loss_fns = ['mse', 'ce', 'normal']
models = ['unet']

for loss_fn, model in product(loss_fns, models):
    os.system(f"python main.py --model {model} --loss_fn {loss_fn} --task train")
    os.system(f"python main.py --model {model} --loss_fn {loss_fn} --task decode --load_ckpt "
              f"ckpt/ae_{loss_fn}_train_cifar10_{model}/best.pth.tar")
    os.system(f"python main.py --model {model} --loss_fn {loss_fn} --task probing --load_ckpt "
              f"ckpt/ae_{loss_fn}_train_cifar10_{model}/best.pth.tar")
    os.system(f"python main.py --model {model} --loss_fn {loss_fn} --task tsne --load_ckpt "
              f"ckpt/ae_{loss_fn}_train_cifar10_{model}/best.pth.tar")