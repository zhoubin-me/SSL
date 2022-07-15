
def main():
    from src.ae.trainer import Trainer
    from src.ae.config import Config

    cfg = Config()
    cfg.task = "Train"
    cfg.dset = "namibia"
    cfg.dset_root = "/ssd1/data/namibia/angles/4.00-55.00-64-lin-angles.tdb"

    print(cfg.__dict__)
    trainer = Trainer(**cfg.__dict__)
    trainer.run()