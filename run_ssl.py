from itertools import product


def main():
    from src.simclr.trainer import Trainer
    from src.simclr.config import Config

    cfg = Config()
    cfg.task = 'train'
    trainer = Trainer(**cfg.__dict__)
    trainer.run()


if __name__ == '__main__':
    main()