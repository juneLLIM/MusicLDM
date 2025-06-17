'''
This code is released and maintained by:

Ke Chen, Yusong Wu, Haohe Liu
MusicLDM: Enhancing Novelty in Text-to-Music Generation Using Beat-Synchronous Mixup Strategies
All rights reserved

contact: knutchen@ucsd.edu
'''
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from latent_diffusion.models.musicldm import MusicLDM
from utilities.data.dataset import BandDataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
import logging
import torch
import yaml
import argparse
import os
from datetime import datetime


def main(args):
    seed_everything(args.seed)

    # Load configuration
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)

    # Cache setup
    cache_path = args.cache_dir
    os.makedirs(cache_path, exist_ok=True)
    os.environ['TRANSFORMERS_CACHE'] = cache_path
    torch.hub.set_dir(cache_path)

    # Logging setup
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(
        args.log_dir, "train.log"), level=logging.INFO)
    log_path = os.path.join(args.log_dir, args.proj_name, args.exp_name)
    print(f'Logs will be saved at {log_path}')

    # Wandb setup
    wandb_logger = WandbLogger(
        save_dir=args.log_dir,
        project=args.proj_name,
        config=config,
        name=args.exp_name,
    )

    # Dataset
    train_dataset = BandDataset(
        args.data_dir, 'train',
        config=config, dataset_iter=10000000
    )
    valid_dataset = BandDataset(
        args.data_dir, 'val',
        config=config, dataset_iter=1
    )

    # Dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Model
    model = MusicLDM(**config["model"]["params"])

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_path, "checkpoints"),
        filename="epoch{epoch}",
        every_n_epochs=args.ckpt_every_n_epochs,
        auto_insert_metric_name=False,
        save_last=True,
        save_top_k=-1
    )

    devices = torch.cuda.device_count()
    trainer = Trainer(
        max_epochs=args.epochs,
        limit_train_batches=args.iterations,
        accelerator="gpu",
        devices=devices,
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=wandb_logger,
        check_val_every_n_epoch=args.val_every_n_epochs,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        default_root_dir=log_path
    )
    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--proj-name",
        type=str,
        required=False,
        default="musicldm"
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        required=False,
        default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=0
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=False,
        default="musicldm.yaml"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=False,
        default="data"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=False,
        default="cache"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        required=False,
        default="log"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=16
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=100
    )
    parser.add_argument(
        "--iterations",
        type=int,
        required=False,
        default=1000
    )
    parser.add_argument(
        "--ckpt-every-n-epochs",
        type=int,
        required=False,
        default=1
    )
    parser.add_argument(
        "--val-every-n-epochs",
        type=int,
        required=False,
        default=1
    )
    args = parser.parse_args()

    main(args)
