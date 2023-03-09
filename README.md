# Deli-CS
This repository contains the code needed to reproduce and explore the results presented in <i>Deep Learning Initialized Compressed Sensing (Deli-CS) in Volumetric Spatio-Temporal Subspace Reconstruction</i> available on: TBC


## Installation.

Run the following commands in sequence to set up your environment to run the experiments. 

1. `conda update -n base -c defaults conda`
2. `make conda`
3. `conda activate deliCS`
4. `make pip`
5. `make data` OR `make data+` (`data` downloads the shared data and testing data (~XXGB), `data+` downloads all training and validation data as well (extra ~50GB))
6. `make docker`

_________________________

Note: the above steps require an Anaconda/Miniconda installation, Docker installation, and nvidia-container-toolkit. 
_________________________

## Pipeline description.
To run the DeliCS pipeline, please navigate to the `pipeline` directory with the `deliCS` conda environment activated. Then run: `python3 XX_script.py` with XX_script.py replaced with the actual script you want to run.

### Data preparation
The data is prepped by ``pipeline/00_prepare.py``. It performs the following steps for all cases in the training, validation and testing
folders where data is available. If the data has already been prepared it will not reprocess the data unless the previously processed data is deleted or renamed. 

1. Using the GRE data: estimate the coil compression matrix with RoVir and calculate shifts for AutoFOV.
2. Prepare acquired MRF data by applying the coil compression, and FOV shifting.
3. Subsample the reference 6min data to 2min when available.
4. Perform initial gridding reconstruction of 2min data and estimate the coil sensitivity maps using JSENSE.
5. Perform reference LLR reconstruction of 6min data when available.
6. Perform reference LLR reconstructions of 2min testing data with varying number of iterations for the test cases.

Reconstruction parameters can be altered using the `pipeline/params.py` file.

Note, since each subject is prepared serially, this step takes a very long time to complete for all training, validation, and test subjects. 

### Block preparation for training
The training and validation data is further processed by ``pipeline/01_make_blocks.py``. Here inputs and targets are augmented and made into blocks to be processed by the DeliCS network during training. The number of augmentations and block size are determined by the `pipeline/params.py` file.

### Train DeliCS
To train DeliCS, run ``pipeline/02_train.py``. To view the progress you can use tensorboard using this command from the main deliCS directory: `tensorboard --logdir logs/case_2min`. This sets up a port that allows you to follow the training progress in your browser.

### Run DeliCS
Once you have a trained network (either by running the pipeline with `data+` or using the provided checkpoints file: `checkpoints/case_2min/version_000/epoch=433-step=276024.ckpt`). The testing data is run through the network using `pipeline/03_deli.py`. This script takes an argument with the path to the file containing the weights. The way you would run it from the pipeline directory is thus: `python3 03_deli.py --chk ../checkpoints/case_2min/version_000/epoch=433-step=276024.ckp`.

The testing blocks differ from training in that they are overlapping and combined using a linear cross-blending method that smooths out the overlapping regions.

### Refinement reconstructons
The refinement reconstruction uses the output of DeliCS as initialization of the same LLR FISTA algorithm that was used for the reference reconstruction in `pipeline/00_prepare.py`-step 6. In the `pipeline/04_refinement.py` script the DeliCS image is first scaled and then reconstructed with varying numbers of iterations to see the convergence of the solution.

### Quantifications
`pipeline/05_quantification.py` is the final step of deliCS. Here the final output is dictionary matched to T1 and T2 values for each voxel in each full and partial reconstruction generated by the pipeline so far.

### Extra - BART reconstruction
For comparison, a script to run a BART reconstruction of one of the training datasets have also been included. `pipeline/recon_2min_bart.py` is not part of the deliCS pipeline, but it generates the reference data presented in the DeliCS paper. It uses the bart installation in the `setsompop/calib` docker container.

## Generate figures from manuscript


## Project structure overview

```
DeliCS
|-- checkpoints
|-- data
|    |-- training
|    |    |-- case000
|    |    |   |-- noise.npy
|    |    |   |-- raw_mrf.npy
|    |    |   |-- raw_gre.npy
|    |    |   |-- ...
|    |    |-- case001
|    |    |   |-- ...
|    |    |   |-- ...
|    |    |-- case002
|    |    |   |-- ...
|    |    |   |-- ...
|    |    |-- ...
|    |    |-- ...
|    |-- validation
|    |    |-- case000
|    |    |   |-- noise.npy
|    |    |   |-- raw_mrf.npy
|    |    |   |-- raw_gre.npy
|    |    |   |-- ...
|    |    |-- case001
|    |    |   |-- ...
|    |    |   |-- ...
|    |-- testing
|    |    |-- case000
|    |    |   |-- noise.npy
|    |    |   |-- raw_mrf.npy
|    |    |   |-- raw_gre.npy
|    |    |   |-- ...
|    |    |-- case001
|    |    |   |-- ...
|    |    |   |-- ...
|    |    |-- case002
|    |    |   |-- ...
|    |    |   |-- ...
|    |    |-- ...
|    |    |-- ...
|    |-- shared
|    |    |-- dcf_2min.npy
|    |    |-- dcf_6min.npy
|    |    |-- dictionary.mat
|    |    |-- phi.mat
|    |    |-- traj_grp16_inacc2.mat
|    |    |-- traj_grp48_inacc1.mat
|-- figures
|    |-- 01_pipeline.txt
|    |-- 02_basis_balancing.ipynb
|    |-- 03_comparebart.ipynb
|    |-- 04_convergence.ipynb
|    |-- 05_compare_recons.ipynb
|    |-- 06_07_block.ipynb
|    |-- 08_compare_recons_patient.ipynb
|    |-- 09_block_patient.ipynb
|    |-- 10_patients.ipynb
|-- logs
|-- MRF [from: https://github.com/SetsompopLab/MRF]
|    |-- src
|    |    |-- 00_io
|    |    |   |-- Dockerfile
|    |    |   |-- main.py
|    |    |   |-- Makefile
|    |    |   |-- ...
|    |    |-- 01_calib
|    |    |   |-- Dockerfile
|    |    |   |-- main.py
|    |    |   |-- Makefile
|    |    |   |-- ...
|    |    |-- 02_recon
|    |    |   |-- Dockerfile
|    |    |   |-- main.py
|    |    |   |-- Makefile
|    |    |   |-- ...
|    |-- README.md
|-- pipeline
|    |-- 00_prepare.py
|    |-- 01_make_blocks.py
|    |-- 02_train.py
|    |-- 03_deli.py
|    |-- 04_estimate_weights.py
|    |-- 05_refinement.py
|    |-- 06_quantification.py
|    |-- metrics.py
|    |-- params.py
|    |-- recon_2min_bart.py
|    |-- resunet.py
|-- environment.yaml
|-- Makefile
|-- README.md
```