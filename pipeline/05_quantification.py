#!/usr/bin/env python3

import os
import numpy as np
import sigpy as sp
from scipy.io import loadmat
import params


dst = os.getcwd()
phi = f"{dst}/../data/shared/phi.mat"
dictionary = f"{dst}/../data/shared/dictionary.mat"

lst_data = [f"../data/testing/{f}" for f in os.listdir("../data/testing")
             if os.path.isfile(f"../data/testing/{f}/refine_{params.accel}_iters_20.npy")]
lst_data = list(set(lst_data))
lst_data.sort()

dst = os.getcwd()

dev = params.devnum


phi = loadmat(phi)["phi"][:, :5]
phi = phi @ sp.fft(np.eye(phi.shape[-1]), axes=(0,))
mat = loadmat(dictionary)

T1 = mat["T1"].ravel()
T2 = mat["T2"].ravel()
mat = mat["I_b"]

(T2, T1) = np.meshgrid(T2, T1)

T1 = T1.ravel()
T2 = T2.ravel()

mat = phi.conj().T @ np.reshape(mat, (-1, mat.shape[-1])).T

nrm = np.linalg.norm(mat, axis=0)
mat = np.array(mat/nrm[None, :],dtype=np.complex64)


def fit(x, dev, mat):
    dev = sp.Device(dev)
    xp = dev.xp
    with dev:
        mat = sp.to_device(mat, dev)
        x = sp.to_device(x, dev)
        
        x     = xp.abs(mat.conj().T @ xp.reshape(x, (5, 256 * 256)))
        lst_idx = sp.to_device(xp.argmax(x, axis=0), -1)
        
        fit_T1  = np.reshape(np.array([T1[idx] for idx in lst_idx], dtype=np.float32), (256, 256))
        fit_T2  = np.reshape(np.array([T2[idx] for idx in lst_idx], dtype=np.float32), (256, 256))
    return (sp.to_device(fit_T1,-1), sp.to_device(fit_T2,-1))



for case in lst_data:
  base = ("%s/%s" % (dst, case))
  for num_iters in [4,6,8,10,12,15,20,25,30,35]:
    recon = "%s/refine_%s_iters_%d.npy" % (base, params.accel, num_iters)
    if os.path.isfile(recon) and not os.path.isfile("%s/T2_refine_%s_iters_%d.npy" % (base, params.accel, num_iters)):
      print("~~~~~~~~> %s" % recon)
      rec = np.load(recon,mmap_mode='r')
      t1 = np.zeros((256,256,256))
      t2 = np.zeros((256,256,256))
      for sl in range(256):
        idx = [slice(None), slice(None, None, -1), sl]
        t1[idx],t2[idx] = fit(rec[tuple(idx+[slice(None)])].T,  dev, mat)
      np.save("%s/T1_refine_%s_iters_%d.npy" % (base, params.accel, num_iters),t1)
      np.save("%s/T2_refine_%s_iters_%d.npy" % (base, params.accel, num_iters),t2)
    
    recon = "%s/uninit_%s_iters_%d.npy" % (base, params.accel, num_iters)
    if os.path.isfile(recon) and not os.path.isfile("%s/T2_uinit_%s_iters_%d.npy" % (base, params.accel, num_iters)):
      print("~~~~~~~~> %s" % recon)
      rec = np.load(recon,mmap_mode='r')
      t1 = np.zeros((256,256,256))
      t2 = np.zeros((256,256,256))
      for sl in range(256):
        idx = [slice(None), slice(None, None, -1), sl]
        t1[idx],t2[idx] = fit(rec[tuple(idx+[slice(None)])].T,  dev, mat)
      np.save("%s/T1_uinit_%s_iters_%d.npy" % (base, params.accel, num_iters),t1)
      np.save("%s/T2_uinit_%s_iters_%d.npy" % (base, params.accel, num_iters),t2)
    
  recon = "%s/ref_%s.npy" % (base, params.accel)
  if os.path.isfile(recon) and not os.path.isfile("%s/T2_ref_%s.npy" % (base, params.accel)):
    print("~~~~~~~~> %s" % recon)
    rec = np.load(recon,mmap_mode='r')
    t1 = np.zeros((256,256,256))
    t2 = np.zeros((256,256,256))
    for sl in range(256):
      idx = [slice(None), slice(None, None, -1), sl]
      t1[idx],t2[idx] = fit(rec[tuple(idx+[slice(None)])].T,  dev, mat)
    np.save("%s/T1_ref_%s.npy" % (base, params.accel),t1)
    np.save("%s/T2_ref_%s.npy" % (base, params.accel),t2)

  recon = "%s/ref_%s.npy" % (base, '6min')
  if os.path.isfile(recon) and not os.path.isfile("%s/T2_ref_%s.npy" % (base, '6min')):
    print("~~~~~~~~> %s" % recon)
    rec = np.load(recon,mmap_mode='r')
    t1 = np.zeros((256,256,256))
    t2 = np.zeros((256,256,256))
    for sl in range(256):
      idx = [slice(None), slice(None, None, -1), sl]
      t1[idx],t2[idx] = fit(rec[tuple(idx+[slice(None)])].T,  dev, mat)
    np.save("%s/T1_ref_%s.npy" % (base, '6min'),t1)
    np.save("%s/T2_ref_%s.npy" % (base, '6min'),t2)



            
