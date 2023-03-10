#!/usr/bin/env python3

import os
import sys
import time
import argparse
from tqdm import tqdm

import numpy as np
import sigpy as sp

# PyTorch
import torch

# PyTorch lightning modules
import pytorch_lightning as pl

import resunet
import params

torch.backends.cudnn.enabled = True
np.random.seed(params.seed)

###############################################################################

class LitResUNet(pl.LightningModule):
  """
  ResUNet model inside a PyTorch Lightning module.
  """

  def __init__(self, data_parallel=False):

    super().__init__()

    self.model = resunet.ResUNet()

  def forward(self, input):
    return self.model(input)

###############################################################################

def block_pass(model, x, device, verbose=False, batch=2):
  # Blocking operator.
  B = sp.linop.ArrayToBlocks(x.shape,
    (params.nn_inf_block_size,)*3,
    (int(params.nn_inf_block_size*(1-params.overlap_fract)),)*3)
  n = int(np.prod(B.oshape[1:4]))
  B = sp.linop.Reshape((params.tk, n) + tuple(B.oshape[-3:]), B.oshape) * B
  B = sp.linop.Transpose(B.oshape, (1, 0, 2, 3, 4)) * B
  if verbose:
    print(f">> Cross-blending {n} blocks.", flush=True)

  # reorder blocks per dimension
  bx, by, bz = (params.nn_inf_block_size,)*3 # block size
  nx = int(np.cbrt(n)) # block grid (only works for equal in all dimensions)
  ny = nx
  nz = nx
  Mz = np.ones([nz, ny, nx, bz, by, bx])
  My = np.ones([nz, ny, nx, bz, by, bx])
  Mx = np.ones([nz, ny, nx, bz, by, bx])
  ox, oy, oz = (params.nn_inf_block_size*params.overlap_fract,)*3 # overlap size

  rise = np.linspace(0, 1, int(oz))
  fall = np.linspace(1, 0, int(oz))


  for ii in range(nz):
    for jj in range(ny):
      for kk in range(ny):
        for nn in range(bz):
          for mm in range(by):
            for oo in range(bx):
              if ii != 0 : # not top edge 
                if nn < oz: # in overlap region (top)
                  Mz[ii,jj,kk,nn,mm,oo] = rise[nn]
              if ii != (nz-1) : # not bottom edge 
                if nn > bz-oz: # in overlap region (bottom)
                  Mz[ii,jj,kk,nn,mm,oo] = fall[int(nn-(bz-oz))]

              if jj != 0 : # not front edge 
                if mm < oy: # in overlap region (front)
                  My[ii,jj,kk,nn,mm,oo] = rise[mm]
              if jj != (ny-1): # not back edge 
                if mm > by-oy: # in overlap region (back)
                  My[ii,jj,kk,nn,mm,oo] = fall[int(mm-(by-oy))]

              if kk != 0 : # not left edge 
                if oo < ox: # in overlap region (left)
                  Mx[ii,jj,kk,nn,mm,oo] = rise[oo]
              if kk != (nx-1): # not right edge
                if oo > bx-ox: # in overlap region (right)
                  Mx[ii,jj,kk,nn,mm,oo] = fall[int(oo-(bx-ox))]


  M = np.reshape(Mx*My*Mz, [-1, bz, by, bx])

  # Extract blocks and prepare arrays.
  scale = np.linalg.norm(x.ravel(), ord=np.inf) + \
            (np.finfo(np.float32).eps)
  x = x/scale
  x = B(x)
  y_shape = list(x.shape)
  y_shape[1] = 2 * params.tk
  y = np.zeros(y_shape, dtype=np.complex64)

  # Forward pass.
  for k in tqdm(range(0, x.shape[0], batch)):
    blk = x[k:k + batch, ...]
    tM = torch.from_numpy(M[k:k+batch, ...]).to(device)
    if len(blk.shape) == 4:
      blk = blk[None, ...]
      tM = tM[None, ...]
    blk[np.isnan(blk)] = 0
    blk[np.isinf(np.abs(blk))] = 0
    blk = np.concatenate((blk.real, blk.imag), axis=1)
    src = torch.as_tensor(blk).to(device)
    res = model(src).squeeze()  * tM[:, None, ...]
    y[k:k + batch, ...] = res.cpu().detach().numpy().squeeze()
  y = y[:, :int(y.shape[1]/2), ...] + 1j * y[:, int(y.shape[1]/2):, ...]
  return B.H(y)

###############################################################################

def main(args):

  if params.nn_inf_block_size < params.N and params.devnum > -1:
    print("> Using CUDA.")
    device = torch.device(f"cuda:{params.devnum}")
  else:
    print("> Using CPU.")
    device = torch.device("cpu")
  
  # Load network.
  print("> Setting up network... ", end="", flush=True)
  checkpoint = os.path.join(args.chk, os.listdir(args.chk).pop(0)) if \
               os.path.isdir(args.chk) else args.chk
  model = LitResUNet.load_from_checkpoint(checkpoint).to(device)
  model.freeze()
  print("done.", flush=True)
  
  lst_data = [f"../data/testing/{f}" for f in os.listdir("../data/testing")
               if os.path.isfile(f"../data/testing/{f}/init_adj_{params.accel}.npy")]
  lst_data = list(set(lst_data))
  lst_data.sort()

  for case in lst_data:
  
    print(("Case: %s" % (case)).capitalize(), flush=True)
    result = f"{case}/deli_{params.accel}.npy"

    if os.path.isfile(result):
      continue
  
    # Load image.
    print("> Loading data... ", end='', flush=True)
    init = np.load(f"{case}/init_adj_{params.accel}.npy").T
    print("done.", flush=True)

    # Forward pass.
    print("> Forward pass:", flush=True)
    start_time = time.perf_counter()
    res = block_pass(model, np.array(init,dtype=np.complex64), device, verbose=True)
    end_time = time.perf_counter()
    print("> Time taken: %fs" % (end_time - start_time), flush=True)

    # Saving data.
    print("> Saving data... ", end="", flush=True)
    np.save(result, res.squeeze().T)
    print("done.", flush=True)


def create_arg_parser():

  parser = argparse.ArgumentParser(description="DELI-CS")
  parser.add_argument('--chk',  type=str, default=None, required=True,
                                help="Load checkpoint")

  return parser


if __name__ == '__main__':
  args = create_arg_parser().parse_args(sys.argv[1:])

  np.random.seed(params.seed)
  torch.manual_seed(params.seed)

  main(args)
