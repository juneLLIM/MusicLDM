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
from src.utilities.data.dataset import TextDataset
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

    # Output setup
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.proj_name, args.exp_name)
    print(f'Results will be saved at {out_path}')

    # Dataset
    dataset = TextDataset(
        data=args.original,
    )

    # Dataloader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Model
    model = MusicLDM(**config["model"]["params"])
    model.eval()
    # model.to(device)

    log_path = os.path.join(args.log_dir, args.proj_name, args.exp_name)
    print(f'Logs will be saved at {log_path}')
    devices = torch.cuda.device_count()
    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        # strategy=DDPStrategy(find_unused_parameters=False),
        # logger=wandb_logger,
        # callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        default_root_dir=log_path
    )
    print(next(iter(loader)))
    trainer.validate(model, loader)
    # model.generate_sample(iter(loader))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proj-name",
        type=str,
        required=False,
        default="musicldm-style-transfer"
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        required=False,
        default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    parser.add_argument(
        "--original",
        type=str,
        required=True,
        help="text or path to the data file to transfer the music sample from",
        nargs='+'
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="text or path to the data file to transfer the music sample to",
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
        "--cache-dir",
        type=str,
        required=False,
        default="cache"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=False,
        default="out"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=16
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        required=False,
        default="log"
    )

    args = parser.parse_args()

    main(args)
