## Reconstruction parameters.
N   = 256  # Matrix size.
tf  = 500  # Number of time points.
ro  = 1688 # Readout length.
rc  = 40   # ROVir projection.
nc  = 10   # SVD coil.
tk  = 5    # Final number of coefficients.
ptt = 10   # Number of readout points to throw away.
accel     = "2min" # How long an MRF scan we retrospectively undersample to (acquired data is either 6min or 2min).
sens_alg  = "jsense" #can change to "espirit" for cleaner maps but longer time.

## GPU device.
devnum = 0
devlst = [devnum]

## Random seed.
seed = 0

## DL parameters
augmentation_n = 16 # Number of augmentations.
nn_block_size = 64           # Block size for training.
nn_inf_block_size = 64       # Block size for inference.
nn_features   = 64           # Number of features in first layer.
nn_kernel     = 3            # Convolution kernel size.
nn_activation = "relu"       # Activation type.
nn_loss       = "complex_l1" # Cost function.
opt_name             = "Adam"
opt_epochs           = 2000
opt_grad_accum_iters = 1
opt_adam_lr          = 1e-5
log_every_n_steps = 25
dataloader_workers = 40
dataloader_batch   = 1
