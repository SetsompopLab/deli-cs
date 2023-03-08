import numpy as np
import sys
from scipy.io import loadmat
import os

# helper functions from https://github.com/mrirecon/bart
def readcfl(name):
    # get dims from .hdr
    h = open(name + ".hdr", "r")
    h.readline() # skip
    l = h.readline()
    h.close()
    dims = [int(i) for i in l.split( )]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n)+1]

    # load data and reshape into dims
    d = open(name + ".cfl", "r")
    a = np.fromfile(d, dtype=np.complex64, count=n);
    d.close()
    return a.reshape(dims, order='F') # column-major

def writecfl(name, array):
    h = open(name + ".hdr", "w")
    h.write('# Dimensions\n')
    for i in (array.shape):
            h.write("%d " % i)
    h.write('\n')
    h.close()
    d = open(name + ".cfl", "w")
    array.T.astype(np.complex64).tofile(d) # tranpose for column-major order
    d.close()

# set up paths
raw_path    =    '../data/training/case000/ksp_2min.npy'
traj_path   =    '../data/shared/traj_grp16_inacc2.mat'
phi_path    =    '../data/shared/phi.mat'
recon_path  =    '../data/training/case000/bartrecon_2min.npy'
mps_path    =    '../data/training/case000/mps_2min.npy'

cwd = os.getcwd()

sens_path_cfl = '../data/training/case000/sens'
phi_path_cfl = '../data/training/case000/phi'
traj_path_cfl = '../data/training/case000/traj'
raw_path_cfl = '../data/training/case000/raw'

# transpose and save data as cfl
raw = np.load(raw_path)
mps = np.load(mps_path).T
traj = loadmat(traj_path)['k_3d'][10:,:,:,:]
traj[:,0,...] = traj[:,0,...] * 256
traj[:,1,...] = traj[:,1,...] * 256
traj[:,2,...] = traj[:,2,...] * 256

phi = loadmat(phi_path)['phi'][:, :5]

phi = np.reshape(phi,(1, 1, 1, 1, 1, 500, 5))
traj = np.reshape(np.transpose(traj,(1,0,2,3)),(3, -1, 8, 1, 1, 500, 1))
raw = np.reshape(np.transpose(raw,(1,2,0,3)),(1, -1, 8, 10, 1, 500,1))
mps = np.reshape(mps,(256,256,256,10,1,1,1))
writecfl(phi_path_cfl,phi)
writecfl(traj_path_cfl,traj)
writecfl(raw_path_cfl,raw)
writecfl(sens_path_cfl,mps)


# hijack the bart installation in the calibration docker to run experiment
cmd1 = f"echo $OMP_NUM_THREADS ; "
cmd2 = f"./bart-0.8.00/bart pics -d 5 -i 300 -s 5e-3 -R L:7:7:5e-5 -B /mnt/{cwd}/{phi_path_cfl} " + \
            f"-t /mnt/{cwd}/{traj_path_cfl} /mnt/{cwd}/{raw_path_cfl} /mnt/{cwd}/{sens_path_cfl} /mnt/{cwd}/{recon_path} "

os.system('docker create --name=bart_container -v /:/mnt/:z -it -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=80 --entrypoint bash setsompop/calib')
os.system('docker start bart_container')
os.system(f"docker exec -it bart_container {cmd1}" )
os.system(f"docker exec -it bart_container {cmd2}" )
os.system('docker stop bart_container ' )
os.system('docker rm bart_container ' )

