import os
import numpy as np
import params

# 0. Set up common parameters and count available data sets.
dst = os.getcwd()
trj = {}
trj["6min"] = f"{dst}/../data/shared/traj_grp48_inacc1.mat"
trj["2min"] = f"{dst}/../data/shared/traj_grp16_inacc2.mat"

eig         = {}
eig["6min"] = 0.0085
eig["2min"] = 0.009

phi = f"{dst}/../data/shared/phi.mat"

os.makedirs("%s/../data/shared" % dst, exist_ok=True)
os.makedirs("%s/../data/training" % dst, exist_ok=True)
os.makedirs("%s/../data/testing" % dst, exist_ok=True)
os.makedirs("%s/../data/validation" % dst, exist_ok=True)

N_train = 0
# Iterate directory
for dir in os.listdir('../data/training'):
    # check if current path is a file
    if os.path.isfile(f'../data/training/{dir}/raw_mrf.npy') and os.path.isfile(f'../data/training/{dir}/raw_gre.npy'):
        N_train += 1
print('Number of training cases:', N_train)

N_val = 0
# Iterate directory
for dir in os.listdir('../data/validation'):
    # check if current path is a file
    if os.path.isfile(f'../data/validation/{dir}/raw_mrf.npy') and os.path.isfile(f'../data/validation/{dir}/raw_gre.npy'):
        N_val += 1
print('Number of validation cases:', N_val)

N_test = 0
# Iterate directory
for dir in os.listdir('../data/testing'):
    # check if current path is a file
    if os.path.isfile(f'../data/testing/{dir}/raw_mrf.npy') and os.path.isfile(f'../data/testing/{dir}/raw_gre.npy'):
        N_test += 1
print('Number of test cases:', N_test)
norms = []

for k in range(N_train+N_val+N_test):
  if k <= N_train-1:
    case = "training"
    idx = k
    isPatient=False
  elif k <= N_train+N_val-1:
    case = "validation"
    idx = k - N_train
    isPatient=False
  elif k <=N_train+N_val+N_test-1:
    case = "testing"
    idx = k - N_train-N_val
    if idx>1: # if patient
      isPatient=True
    else:
      isPatient=False
  else:
    raise Exception("Please change data partitioning")


  base = "%s/../data/%s/case%03d" % (dst, case, idx)
  os.makedirs(base, exist_ok=True)

  print("=" * 50)
  

  # 1. Coil compression matrix with RoVir + AutoFOV.
  if not os.path.isfile("%s/ccm.npy" % (base)) and not os.path.isfile("%s/shifts.npy" % (base)):
    os.system("docker run --gpus all -v /:/mnt/:z setsompop/calib " + \
              ("--ksp /mnt/%s/raw_gre.npy " % (base))              + \
              ("--nse /mnt/%s/noise.npy " % (base))                + \
              ("--ccm /mnt/%s/ccm.npy " % (base))                  + \
              ("--shf /mnt/%s/shifts.npy " % (base))               + \
              ("--nrc %d --nsv %d" % (params.rc, params.nc)))

  # 2. Prepare reference k-space (6min for healthy volunteers and 2min for patients). - Apply coil compression and FOV shifting.
  if not os.path.isfile("%s/ksp_6min.npy" % (base)) and not isPatient:
    os.system("docker run --gpus all -v /:/mnt/:z setsompop/recon -a " +     \
              ("--trj /mnt/%s " % (trj["6min"])) +                          \
              ("--ksp /mnt/%s/raw_mrf.npy " % (base)) +                     \
              ("--res /mnt/%s/tmp.npy " % (base)) +                         \
              ("--phi /mnt/%s " % (phi)) +                                  \
              ("--ccm /mnt/%s/ccm.npy " % (base)) +                         \
              ("--shf /mnt/%s/shifts.npy " % (base)) +                      \
              ("--svk /mnt/%s/ksp_6min.npy " % (base)) +                    \
              ("--mal %s " % (params.sens_alg)) +                      \
              ("--mtx %d --ptt %d --dev %d " % \
                (params.N, params.ptt, params.devnum)))

  if not os.path.isfile("%s/ksp_2min.npy" % (base)) and isPatient:
    os.system("docker run --gpus all -v /:/mnt/:z setsompop/recon -a " +     \
              ("--trj /mnt/%s " % (trj["2min"])) +                          \
              ("--ksp /mnt/%s/raw_mrf.npy " % (base)) +                     \
              ("--res /mnt/%s/tmp.npy " % (base)) +                         \
              ("--phi /mnt/%s " % (phi)) +                                  \
              ("--ccm /mnt/%s/ccm.npy " % (base)) +                         \
              ("--shf /mnt/%s/shifts.npy " % (base)) +                      \
              ("--svk /mnt/%s/ksp_2min.npy " % (base)) +                    \
              ("--mal %s " % (params.sens_alg)) +                      \
              ("--mtx %d --ptt %d --dev %d " % \
                (params.N, params.ptt, params.devnum)))

  if os.path.isfile("%s/tmp.npy" % (base)):
    os.system("rm -f %s/tmp.npy" % (base))


  # 3. Sub-sampling.
  if params.accel == "2min" and not os.path.isfile("%s/ksp_2min.npy" % (base)) and not isPatient:
    ksp = np.load("%s/ksp_6min.npy" % (base), mmap_mode="r")

    ksp_a = ksp[:, :, 0:32:2, 0:500:2]
    ksp_b = ksp[:, :, 1:32:2, 1:500:2]

    ksp = np.zeros(list(ksp_a.shape[:-1]) + [ksp.shape[-1]], dtype=ksp.dtype)
    ksp[..., 0:500:2] = ksp_a
    ksp[..., 1:500:2] = ksp_b

    np.save("%s/ksp_2min.npy" % (base), ksp)

  # 4. Gridding reconstruction (2-min).
  recon = ("%s/init_adj_%s.npy" % (base, params.accel))
  if not os.path.isfile(recon):
    print("~~~~~~~~> %s" % recon,flush=True)
    os.system("docker run --gpus all  " + \
              ("-v /:/mnt/:z setsompop/recon -a ") +                                \
              ("--trj /mnt/%s " % (trj[params.accel])) +                           \
              ("--ksp /mnt/%s/ksp_%s.npy " % (base, params.accel)) +               \
              ("--dcf /mnt/%s/../data/shared/dcf_%s.npy " % (dst, params.accel)) +    \
              ("--mps /mnt/%s/mps_%s.npy " % (base, params.accel)) +                           \
              ("--mal %s " % (params.sens_alg)) +                             \
              ("--phi /mnt/%s " % (phi)) +                                         \
              ("--res /mnt/%s " % (recon)) +                                       \
              ("--akp --mtx %d --ptt %d --dev %d " % (params.N, params.ptt,
                                                      params.devnum)))
                                                        
  # 5. Reference 6-min reconstruction.
  if not os.path.isfile("%s/ref_6min.npy" % (base)) and not isPatient and case=="testing":
    print("~~~~~~~~> %s/ref_6min.npy" % base,flush=True)
    os.system("docker run --gpus all " + \
      ("-v /:/mnt/:z setsompop/recon -p ") +                                        \
      ("--trj /mnt/%s " % (trj["6min"])) +                                         \
      ("--ksp /mnt/%s/ksp_6min.npy " % (base)) +                                   \
      ("--dcf /mnt/%s/../data/shared/dcf_6min.npy " % (dst)) +                        \
      ("--mps /mnt/%s/mps_%s.npy " % (base, params.accel)) +                                   \
      ("--mal %s " % (params.sens_alg)) +                                     \
      ("--res /mnt/%s/ref_6min.npy " % (base)) +                                   \
      ("--phi /mnt/%s " % (phi)) +                                                 \
      ("--eig %f " % (eig["6min"])) +                                         \
      ("--pdg 0 --blk 8 --lam 3e-5 --mit 40 ") +                              \
      ("--akp --mtx %d --ptt %d --dev %d" % (params.N,
                                              params.ptt,
                                              params.devnum)))
    
  # 4h. Reference 2-min reconstruction.
  recon = ("%s/ref_%s.npy" % (base, params.accel))
  if not os.path.isfile(recon):
    print("~~~~~~~~> %s" % recon,flush=True)
    os.system("docker run --gpus all " + \
      ("-v /:/mnt/:z setsompop/recon -p ") +                                        \
      ("--trj /mnt/%s " % (trj[params.accel])) +                                   \
      ("--ksp /mnt/%s/ksp_%s.npy " % (base, params.accel)) +                       \
      ("--dcf /mnt/%s/../data/shared/dcf_%s.npy " % (dst, params.accel)) +            \
      ("--mps /mnt/%s/mps_%s.npy " % (base, params.accel)) +                                   \
      ("--mal %s " % (params.sens_alg)) +                             \
      ("--res /mnt/%s " % (recon)) +                                               \
      ("--phi /mnt/%s " % (phi)) +                                                 \
      ("--eig %f " % (eig[params.accel])) +                                   \
      ("--pdg 0 --blk 8 --lam 5e-5 --mit 40 ") +                              \
      ("--akp --mtx %d --ptt %d --dev %d" % (params.N,
                                            params.ptt,
                                            params.devnum)))

  for num_iters in [4,6,8,10,12,15,20,25,30,35]:
    recon = "%s/uninit_%s_iters_%d.npy" % (base, params.accel, num_iters)
    if not os.path.isfile(recon) and not isPatient and case=="testing":
      print("~~~~~~~~> %s" % recon)
      os.system("docker run --gpus all "+\
        ("-v /:/mnt/:z setsompop/recon -p ") +                                      \
        ("--trj /mnt/%s " % (trj[params.accel])) +                                 \
        ("--ksp /mnt/%s/ksp_%s.npy " % (base, params.accel)) +                     \
        ("--dcf /mnt/%s/../data/shared/dcf_%s.npy " % (dst, params.accel)) +          \
        ("--mps /mnt/%s/mps_%s.npy " % (base, params.accel)) +                                      \
        ("--mal %s " % (params.sens_alg)) +                             \
        ("--res /mnt/%s " % (recon)) +                                             \
        ("--phi /mnt/%s " % (phi)) +                                               \
        ("--eig %f " % (eig[params.accel])) +                                 \
        ("--pdg 0 --blk 8 --lam 5e-5 --mit %d " % (num_iters)) +              \
        ("--akp --mtx %d --ptt %d --dev %d" % (params.N,
                                                params.ptt,
                                                params.devnum)))

# 5. When all 10 training subjects are available calculate the average energy in the reference image to scale the deliCS input
  recon = ("%s/ref_%s.npy" % (base, params.accel))
  if os.path.isfile(recon) and case=="training" and N_train==10:
    x = np.load(("%s/ref_%s.npy" % (base, params.accel)), mmap_mode="r")
    norms.append(np.linalg.norm(x))
    print(f'Norm for ref-2min for case: {case}/{idx} is: {norms[-1]}')

if N_train==10:
  val = np.mean(norms)
  print(f"Estimated norm-scaling ({params.accel}):", val)
  np.save(f"../data/shared/deli_scaling_{params.accel}.npy", val)
