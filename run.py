from itertools import product


def main():
    algos = ['vae']
    loss_fns = ['mse', 'ce', 'normal', 'gmm']
    models = ['unet', 'resnet', 'shallow_net', 'unet_no_pyr']
    few_perc = [1.0, 0.002]

    for loss_fn, model, algo, perc in product(loss_fns, models, algos, few_perc):
        if algo == 'ae':
            from src.ae.config import Config
            from src.ae.trainer import Trainer
        elif algo == 'vae':
            from src.vae.config import Config
            from src.vae.trainer import Trainer
        else:
            raise ValueError("No such algo")

        cfg = Config()
        cfg.few_perc = perc
        if perc == 1.0:
            cfg.prefix = f'{algo}'
        else:
            cfg.prefix = f'{algo}_{perc}'

        cfg.model = model
        cfg.loss_fn = loss_fn
        cfg.load_ckpt = f"ckpt/{algo}_{loss_fn}_train_cifar10_{model}/best.pth.tar"

        cfg.task = "train"
        print(cfg.__dict__, algo)
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

if __name__ == '__main__':
    main()