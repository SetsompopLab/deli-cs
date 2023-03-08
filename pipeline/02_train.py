#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np

# PyTorch
import torch
from torch.utils.data import DataLoader, Dataset

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import metrics as mt
import params

import resunet


class LitResUnet(pl.LightningModule):
  """
  ResUNet model inside a PyTorch Lightning module.
  """

  def __init__(self, data_parallel=False):

    super().__init__()

    self.model = resunet.ResUNet()


  def compute_metrics(self, prediction, target, is_training=True):
  	# Initialize dictionary of metrics
    metrics = {}
    tag = 'Train' if is_training else 'Validate'

    # Entry-based error metrics
    metrics[f'{tag}/l1']   = mt.l1  (target, prediction)
    metrics[f'{tag}/l2']   = mt.l2  (target, prediction)
    metrics[f'{tag}/psnr'] = mt.psnr(target, prediction)

    # Get magnitudes of each tensor
    nc = int(prediction.shape[1]/2)
    cPrediction = prediction[:, :nc, ...] + 1j * prediction[:, nc:, ...]
    cTarget     = target[:, :nc, ...]     + 1j * target[:, nc:, ...]

    # Complex-based error metrics
    metrics[f'{tag}/complex_l1']   = mt.l1  (cTarget, cPrediction)
    metrics[f'{tag}/complex_l2']   = mt.l2  (cTarget, cPrediction)
    metrics[f'{tag}/complex_psnr'] = mt.psnr(cTarget, cPrediction)

    # Magnitude-based error metrics
    metrics[f'{tag}/magn_l1']   = mt.l1  (cTarget.abs(), cPrediction.abs())
    metrics[f'{tag}/magn_l2']   = mt.l2  (cTarget.abs(), cPrediction.abs())
    metrics[f'{tag}/magn_psnr'] = mt.psnr(cTarget.abs(), cPrediction.abs())

    return metrics


  def train_dataloader(self):

    class tds(Dataset):

      def __init__(self):
        base = f"data/training/blocks_{params.accel}"
        self.locs = [("%s/%s" % (base, f)) for f in os.listdir(base)
                     if f.startswith("blk")]

      def __len__(self):
        return len(self.locs)

      def __getitem__(self, index):
        nc = 2 * params.tk
        arr = np.load(self.locs[index])
        (src, ref) = (arr[:nc, ...], arr[nc:, ...])
        return [torch.as_tensor(x) for x in (src, ref)]

    tds = tds()

    loader = DataLoader(dataset=tds, batch_size=params.dataloader_batch,
                        num_workers=params.dataloader_workers, pin_memory=True,
                        shuffle=True)

    return loader


  def val_dataloader(self):

    class tds(Dataset):

      def __init__(self):
        base = f"data/validation/blocks_{params.accel}"
        self.locs = [("%s/%s" % (base, f)) for f in os.listdir(base)
                     if f.startswith("blk")]

      def __len__(self):
        return len(self.locs)

      def __getitem__(self, index):
        nc = 2 * params.tk
        arr = np.load(self.locs[index])
        (src, ref) = (arr[:nc, ...], arr[nc:, ...])
        return [torch.as_tensor(x) for x in (src, ref)]

    tds = tds()

    loader = DataLoader(dataset=tds, batch_size=params.dataloader_batch,
                        num_workers=params.dataloader_workers, pin_memory=True)

    return loader


  def log_data(self, prediction, target):

    # Helper function for logging images
    def save_image(image, tag):
      x = image/image.max()
      self.logger.experiment.add_image(tag, x[None, ...],
                                       global_step=self.global_step)

    nc = int(prediction.shape[1]/2)
    sz = int(prediction.shape[2]/2)

    cPrediction = prediction[0, :nc, sz, :, :] + 1j * \
                  prediction[0, nc:, sz, :, :]
    cTarget     = target[0, :nc, sz, :, :] + 1j * \
                  target[0, nc:, sz, :, :]

    # Stack images from left-to-right.
    errors = torch.abs(cPrediction - cTarget)
    images = torch.abs(torch.cat((cPrediction, cTarget, errors), dim=-1))

    for e in range(images.shape[0]):
      save_image(images[e, ...], "Coefficient %d: Pred / Target / Error" % (e))


  def training_step(self, batch, batch_idx):

    # Load batch of input-output pairs
    (src, ref) = batch

    pred    = self.model(src)
    metrics = self.compute_metrics(pred, ref, is_training=True)
    self.log_dict(metrics)

    if (self.global_step + 1) % params.log_every_n_steps == 0:
      self.log_data(pred, ref)

    return metrics[f'Train/{params.nn_loss}']


  def validation_step(self, batch):

    # Load batch of input-output pairs
    (src, ref) = batch

    # Perform forward pass through unrolled network
    pred = self.model(src)

    # Compute and log metrics
    metrics = self.compute_metrics(pred, ref, is_training=False)
    self.log_dict(metrics)


  def on_train_start(self):
    self.logger.log_hyperparams(
      params={"batch": params.dataloader_batch,
              "num_augmentations": params.augmentation_n,
              "nn_block_size": params.nn_block_size,
              "nn_res_blocks": params.nn_res_blocks,
              "nn_features": params.nn_features,
              "nn_kernel": params.nn_kernel,
              "nn_activation": params.nn_activation,
              "nn_loss": params.nn_loss,
              "opt_name": params.opt_name,
              "opt_epochs": params.opt_epochs,
              "opt_grad_accum_iters": params.opt_grad_accum_iters,
              "opt_adam_lr": params.opt_adam_lr},
      metrics={"Validate/complex_l1": 0.1,
               "Validate/complex_l2": 0.1,
               "Validate/complex_psnr": 10})


  def configure_callbacks(self):
    os.makedirs(f"checkpoints/case_{params.accel}to{params.target_accel}", exist_ok=True)

    lst = os.listdir(f"checkpoints/case_{params.accel}to{params.target_accel}")
    if len(lst) == 0:
      loc = "version_%03d" % 0
    else:
      lst.sort()
      loc = "version_%03d" % (int(lst[-1].split("_")[-1]) + 1)

    os.mkdir(f"checkpoints/case_{params.accel}to{params.target_accel}/{loc}")
    checkpoint = ModelCheckpoint(
      dirpath=f"checkpoints/case_{params.accel}{params.target_accel}/{loc}",
      save_top_k=1,
      monitor=f'Validate/{params.nn_loss}',
      mode='min',
      verbose=True
    )
    return [checkpoint]


  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=params.opt_adam_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=1000,
                                                gamma=0.5)
    return [optimizer], [scheduler]


def main(args):

  # Initialize unrolled model
  model = LitResUnet() if args.chk is None else \
          LitResUnet.load_from_checkpoint(args.chk)

  # Initialize logger (for writing summary to TensorBoard)
  logger = TensorBoardLogger(save_dir="logs",
                             name=f"case_{params.accel}",
                             default_hp_metric=False)

  # Initialize a trainer
  trainer = Trainer(accelerator='gpu', devices=params.devlst,
                    default_root_dir="checkpoints",
                    logger=logger,
                    max_epochs=params.opt_epochs,
                    log_every_n_steps=params.log_every_n_steps,
                    check_val_every_n_epoch=1,
                    accumulate_grad_batches=params.opt_grad_accum_iters)

  # Train the model
  trainer.fit(model)


def create_arg_parser():

  parser = argparse.ArgumentParser(description="DELICS TRAIN")
  parser.add_argument('--chk',  type=str, default=None, required=False,
                                help="Load checkpoint")

  return parser


if __name__ == '__main__':
  args = create_arg_parser().parse_args(sys.argv[1:])

  # Parse command line arguments
  print("DeliCS Training")

  # Set random seeds
  np.random.seed(params.seed)
  torch.manual_seed(params.seed)

  main(args)
