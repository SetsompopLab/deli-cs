# Deli-CS
This repository contains the code needed to reproduce and explore the results presented in <i>Deep Learning Initialized Compressed Sensing (Deli-CS) in Volumetric Spatio-Temporal Subspace Reconstruction</i> available on: TBC


## Installation.

Run the following commands in sequence to set up your environment to run the experiments. 
_________________________

Requires an Anaconda/Miniconda and Docker installation as well as nvidia-container-toolkit (`sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit-base`)

_________________________


1. `conda update -n base -c defaults conda`
2. `make conda`
3. `conda activate deliCS`
4. `make pip`
5. `make data` OR `make data+` (`data` downloads the shared data and testing data, `data+` downloads all training data as well (extra ~50GB))
6. `make docker`

## Pipeline description.
### Data preparation
The data is prepped by ``pipeline/00_prepare.py``. It performs the following steps for all cases in the training, validation and testing
folders where data is available. If the data has already been prepared it will not reprocess the data unless the previously processed data is deleted or renamed.

1. Using the GRE data: estimate the coil compression matrix with RoVir and calculate shifts for AutoFOV.
2. Prepare acquired MRF data by applying the coil compression, FOV shifting, and estimate the coil sensitivity maps using JSENSE.
3. Subsample the reference 6min data to 2min when available.
4. Perform initial gridding reconstruction of 2min data.
5. Perform reference LLR reconstruction of 6min data when available.
6. Perform reference LLR reconstructions of 2min testing data with varying number of iterations for the test cases.

### Block preparation for training
The training and validation data is further processed by ``pipeline/01_make_blocks.py``. Here inputs and targets are augmented and made into blocks to be processed by the DeliCS network during training.

### Train DeliCS
To train DeliCS, run ``pipeline/02_train.py``. To view the progress you can use tensorboard.

### Refinement reconstructons

### Quantifications

## Generate figures from manuscript


## Project structure overview

```
DeliCS
|-- data
|    |-- training
|    |    |-- case000
|    |    |   |-- raw_mrf.npy
|    |    |   |-- raw_gre.npy
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
|    |    |   |-- raw_mrf.npy
|    |    |   |-- raw_gre.npy
|    |    |-- case001
|    |    |   |-- ...
|    |    |   |-- ...
|    |-- testing
|    |    |-- case001
|    |    |   |-- raw_mrf.npy
|    |    |   |-- raw_gre.npy
|    |    |-- case002
|    |    |   |-- ...
|    |    |   |-- ...
|    |    |-- case003
|    |    |   |-- ...
|    |    |   |-- ...
|    |    |-- ...
|    |    |-- ...
|    |-- shared
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
|    |-- resunet.py
|-- environment.yaml
|-- Makefile
|-- README.md
```