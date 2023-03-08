## Dimensions.
N   = 256  # Matrix size.
tf  = 500  # Number of time points.
ro  = 1688 # Readout length.
rc  = 40   # ROVir projection.
nc  = 10   # SVD coil.
tk  = 5    # Final number of coefficients.
ptt = 10   # Number of readout points to throw away.

## Experiment parameters.
accel     = "2min" 
fast_inf  = False
sens_alg  = "jsense" #can change to "espirit" for cleaner maps but longer time.

## Device.
devnum = 0
devlst = [devnum]

## Random seed.
seed = 0

## Data loader parameters.
dataloader_workers = 40
dataloader_batch   = 1

## TODO
augmentation_n = 16 # Number of augmentations.

## ResNet parameters.
nn_block_size = 64           # Block size for training.
nn_inf_block_size = 64       # Block size for inference.
nn_features   = 128          # Number of features.
nn_kernel     = 3            # Convolution kernel size.
nn_activation = "relu"       # Activation type.
nn_loss       = "complex_l1" # Cost function.

## Optimizer parameters.
opt_name             = "Adam"
opt_epochs           = 2000
opt_grad_accum_iters = 1
opt_adam_lr          = 1e-5

## Logging parameters.
log_every_n_steps = 25
