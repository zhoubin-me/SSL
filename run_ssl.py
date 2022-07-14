from itertools import product


def main():
    from src.simclr.trainer import Trainer
    from src.simclr.config import Config

    cfg = Config()
    cfg.load_ckpt = f"ckpt/simclr_train_cifar10/best.pth.tar"
    cfg.task = 'probing'
    trainer = Trainer(**cfg.__dict__)
    trainer.run()


if __name__ == '__main__':
    main()