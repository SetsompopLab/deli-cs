#!/usr/bin/env python3

import os
import numpy as np
import params

# 0. Common parameters.
dst = os.getcwd()
trj = {}
trj["6min"] = f"{dst}/../data/shared/traj_grp48_inacc1.mat"
trj["2min"] = f"{dst}/../data/shared/traj_grp16_inacc2.mat"

eig         = {}
eig["6min"] = 0.0085
eig["2min"] = 0.009

phi = f"{dst}/../data/shared/phi.mat"

lst_data_healthies = [f"../data/testing/{f}" for f in os.listdir("../data/testing")
             if os.path.isfile(f"../data/testing/{f}/ref_6min.npy")]
lst_data = list(set(lst_data_healthies))
lst_data.sort()

dst = os.getcwd()
for case in lst_data:
  base = ("%s/%s" % (dst, case))

  # Scale initialization.
  recon = "%s/deli_%s_scaled.npy" % (base, params.accel)
  if not os.path.isfile(recon):
    scaling = np.load(f"../data/shared/deli_scaling_{params.accel}.npy")
    init = np.load("%s/deli_%s.npy" % (base, params.accel)).T
    init = scaling * init/np.linalg.norm(init)
    np.save(recon, init.T)


  for num_iters in [4,6,8,10,12,15,20,25,30,35,40]:
    recon = "%s/refine_%s_iters_%d.npy" % (base, params.accel, num_iters)
    if not os.path.isfile(recon):
      print("~~~~~~~~> %s" % recon)
      os.system("docker run --gpus all "+\
        ("-v /:/mnt/:z setsompop/recon -p ") +                                      \
        ("--trj /mnt/%s " % (trj[params.accel])) +                                 \
        ("--ksp /mnt/%s/ksp_%s.npy " % (base, params.accel)) +                     \
        ("--dcf /mnt/%s/../data/shared/dcf_%s.npy " % (dst, params.accel)) +          \
        ("--int /mnt/%s/deli_%s_scaled.npy " % (base, params.accel)) +                   \
        ("--mps /mnt/%s/mps_%s.npy " % (base,params.accel)) +                                      \
        ("--res /mnt/%s " % (recon)) +                                             \
        ("--phi /mnt/%s " % (phi)) +                                               \
        ("--eig %f " % (eig[params.accel])) +                                 \
        ("--pdg 0 --blk 8 --lam 5e-5 --mit %d " % (num_iters)) +              \
        ("--akp --mtx %d --ptt %d --dev %d" % (params.N,
                                                params.ptt,
                                                params.devnum)))
      
lst_data_patients = [f"../data/testing/{f}" for f in os.listdir("../data/testing")
             if not os.path.isfile(f"../data/testing/{f}/ref_6min.npy")]
lst_data = list(set(lst_data_patients))
lst_data.sort()

dst = os.getcwd()
for case in lst_data:
  base = ("%s/%s" % (dst, case))

  # Scale initialization.
  recon = "%s/deli_%s_scaled.npy" % (base, params.accel)
  if not os.path.isfile(recon):
    scaling = np.load(f"../data/shared/deli_scaling_{params.accel}.npy")
    init = np.load("%s/deli_%s.npy" % (base, params.accel)).T
    init = scaling * init/np.linalg.norm(init)
    np.save(recon, init.T)


  for num_iters in [20,]:
    recon = "%s/refine_%s_iters_%d.npy" % (base, params.accel, num_iters)
    if not os.path.isfile(recon):
      print("~~~~~~~~> %s" % recon)
      os.system("docker run --gpus all "+\
        ("-v /:/mnt/:z setsompop/recon -p ") +                                      \
        ("--trj /mnt/%s " % (trj[params.accel])) +                                 \
        ("--ksp /mnt/%s/ksp_%s.npy " % (base, params.accel)) +                     \
        ("--dcf /mnt/%s/../data/shared/dcf_%s.npy " % (dst, params.accel)) +          \
        ("--int /mnt/%s/deli_%s_scaled.npy " % (base, params.accel)) +                   \
        ("--mps /mnt/%s/mps_%s.npy " % (base,params.accel)) +                                      \
        ("--res /mnt/%s " % (recon)) +                                             \
        ("--phi /mnt/%s " % (phi)) +                                               \
        ("--eig %f " % (eig[params.accel])) +                                 \
        ("--pdg 0 --blk 8 --lam 5e-5 --mit %d " % (num_iters)) +              \
        ("--akp --mtx %d --ptt %d --dev %d" % (params.N,
                                                params.ptt,
                                                params.devnum)))

