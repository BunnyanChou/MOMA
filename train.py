import argparse
import math
import os
import pdb
import pprint
from distutils.util import strtobool
from pathlib import Path
import torch
import numpy as np
import pytorch_lightning as pl
from loguru import logger as loguru_logger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin, DDPShardedPlugin
from pytorch_lightning.utilities import rank_zero_only

from src.config.default import get_cfg_defaults
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_moma import PL_MOMA
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler

loguru_logger = get_rank_zero_only_logger(loguru_logger)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def parse_args():
    # init a custom parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("data_cfg_path", default="./configs/data/megadepth_trainval_640.py", type=str, help="data config path")
    parser.add_argument("main_cfg_path", default="./configs/loftr/outdoor/loftr_ds_dense.py", type=str, help="main config path")
    parser.add_argument("--exp_name", type=str, default="default_exp_name")
    parser.add_argument("--batch_size", type=int, default=1, help="batch_size per gpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--pin_memory",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        default=True,
        help="whether loading data to pinned memory or not",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="pretrained checkpoint path, helpful for using a pre-trained coarse-only model",
    )
    parser.add_argument(
        "--disable_ckpt",
        action="store_true",
        help="disable checkpoint saving (useful for debugging).",
    )
    parser.add_argument(
        "--profiler_name",
        type=str,
        default=None,
        help="options: [inference, pytorch], or leave it unset",
    )
    parser.add_argument(
        "--parallel_load_data",
        action="store_true",
        help="load datasets in with multiple processes.",
    )

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    # pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    # TODO: Use different seeds for each dataloader workers
    # This is needed for data augmentation

    logger = TensorBoardLogger(
        save_dir="./OUTPUT/densematching",
        name=args.exp_name,
        default_hp_metric=False,
    )
    ckpt_dir = Path(logger.log_dir) / "checkpoints"

    last_ckpt_path = str(
        Path(logger.log_dir[:-1] + str(int(logger.log_dir[-1]) - 1))
        / "checkpoints/last.ckpt"
    )
    # if os.path.exists(last_ckpt_path):
    #     args.ckpt_path = last_ckpt_path
    if args.ckpt_path is None:
        pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    else:
        pl.seed_everything(np.random.randint(2**31))  # reproducibility

    # scale lr and warmup-step automatically
    # pdb.set_trace()
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_MOMA(config, pretrained_ckpt=args.ckpt_path, profiler=profiler)
    loguru_logger.info(f"MOMA LightningModule initialized!")

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"MOMA DataModule initialized!")

    # Callbacks
    # TODO: update ModelCheckpoint to monitor multiple metrics
    ckpt_callback = ModelCheckpoint(
        monitor="auc@10",
        verbose=True,
        save_top_k=5,
        mode="max",
        save_last=True,
        dirpath=str(ckpt_dir),
        filename="{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)

    # Lightning Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        plugins=DDPPlugin(
            find_unused_parameters=False,  # True,
            num_nodes=args.num_nodes,
            # strategy="ddp_sharded",
            sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
        ),
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
        replace_sampler_ddp=False,  # use custom sampler
        reload_dataloaders_every_epoch=False,  # avoid repeated samples!
        weights_summary="full",
        # resume_from_checkpoint=args.ckpt_path,
        profiler=profiler,
        # precision=16,
        # auto_lr_find=True
    )
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")

    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()
