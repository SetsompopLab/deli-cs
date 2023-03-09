#!/usr/bin/env python3

import os
import numpy as np
import sigpy as sp
import params

np.random.seed(params.seed)

lst_data =  [f"../data/training/{f}" for f in os.listdir("../data/training")
             if os.path.isfile(f"../data/training/{f}/ref_{params.accel}.npy")]
lst_data += [f"../data/validation/{f}" for f in os.listdir("../data/validation")
             if os.path.isfile(f"../data/validation/{f}/ref_{params.accel}.npy")]
lst_data = list(set(lst_data))
lst_data.sort()

B = sp.linop.ArrayToBlocks((params.tk, params.N, params.N, params.N),
                           [params.nn_block_size]*3,
                           [params.nn_block_size]*3)
n = int(np.prod(B.oshape)/(params.tk * params.nn_block_size**3))
R = sp.linop.Reshape((params.tk, n, params.nn_block_size,
                                    params.nn_block_size,
                                    params.nn_block_size), B.oshape)
T = sp.linop.Transpose(R.oshape, (1, 0, 2, 3, 4))
B = T * R * B

N = int(params.tk/2)

block_index = 0
for case in lst_data:

  print(("Case: %s" % (case)).capitalize(), flush=True)
  base = case.split("/")[2]
  save = f"../data/{base}/blocks_{params.accel}"

  os.makedirs(save, exist_ok=True)

  src = np.load(f"{case}/init_adj_{params.accel}.npy").T
  dst = np.load(f"{case}/ref_{params.accel}.npy").T

  src = src/np.linalg.norm(src[int(params.tk/2), ...])
  dst = dst/np.linalg.norm(dst[int(params.tk/2), ...])

  rng = lambda k: np.random.randint(k)
  for aug_idx in range(params.augmentation_n):

    # Random shift.
    rand_shift = [rng(sdx) for sdx in src.shape[1:]]
    x = np.roll(src, rand_shift, (1, 2, 3))
    y = np.roll(dst, rand_shift, (1, 2, 3))

    # Random flip.
    rand_flip = [(-1)**rng(2) for _ in range(3)]
    x = x[:, ::rand_flip[0], ::rand_flip[1], ::rand_flip[2]]
    y = y[:, ::rand_flip[0], ::rand_flip[1], ::rand_flip[2]]

    # Random transpose.
    rand_perm = [0] + list(np.random.permutation(3) + 1)
    x = np.transpose(x, rand_perm)
    y = np.transpose(y, rand_perm)

    # Blocking.
    x = B(x)
    y = B(y)

    # Number of blocks.
    num_blocks = x.shape[0]
    assert num_blocks == y.shape[0]

    # Saving blocks
    for blk_dx in range(num_blocks):
      x_blk = x[blk_dx, ...]
      y_blk = y[blk_dx, ...]

      # Random scaling between 0.5 and 1.
      scale = np.random.rand()/2 + 0.5 
      scale_x = np.linalg.norm(x_blk[N, ...].ravel(), ord=np.inf)
      scale_y = np.linalg.norm(y_blk[N, ...].ravel(), ord=np.inf)
      # Discard empty blocks.
      if scale_x < 1e-12 or scale_y < 1e-12:
        continue  
      if np.std(y_blk[N,...].ravel()) < 0.3*scale_y:
        continue
      x_blk = x_blk /  scale_x * scale
      y_blk = y_blk /  scale_x * scale

      # Splitting real and imaginary
      x_blk = np.concatenate((x_blk.real, x_blk.imag), axis=0)
      y_blk = np.concatenate((y_blk.real, y_blk.imag), axis=0)

      # Final block.
      blk = np.concatenate((x_blk, y_blk), axis=0)
      print("     Saving: %s/blk_%08d.npy" % (save, block_index),flush=True)
      np.save("%s/blk_%08d.npy" % (save, block_index), np.array(blk, dtype=np.float32), allow_pickle=False)
      block_index = block_index + 1
