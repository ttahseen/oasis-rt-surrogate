# OASIS-RT Surrogate

## About

This repository contains code to generate a surrogate model for the Radiative Transfer component of the OASIS GCM, specifically for simulations of Venus.
Shortwave (sw) and longwave (lw) regimes have been emulated separately, and emulators have been created on two levels:
  1. Emulating the entire schema: inputs = dynamical inputs (P, T, $\rho$)
  2. Emulating the second step of the RT computation: inputs = optical inputs ($\tau_{ext}$, $\tau_{ray}$, $\tau$) 

## Repository Structure

- `model`: Contains model training scripts
  - `rnn_sw`:
      - `dynamical`
      - `optical`
  - `rnn_lw`
- `trained_models`: Contains trained models
- `data`:
  - `raw_data`: Contains raw outputs of OASIS-RT and OASIS grid
  - `preprocessed_data`: Contains preprocessed data for both LW and SW schemas
    - `rnn_sw`
    - `rnn_lw`
  - `opacities_data`: Contains intermediate processing of opacities data for SW schema
  - `constants`: Contains scaling factors used during data preprocessing
- `ifile`: Contains OASIS physical inputs
- `jobscripts`: Contains jobscripts written for Hypatia
- `analysis`
  - `raw-data`: Contains Jupyter NBs for inspecting raw simulation output to inform data preprocessing
  - `preprocessed-data`: Contains Jupyter NBs for inspecting the preprocessed data to check it has been processed as intended
  - `trained-models`: Contains scripts for loading models from checkpoints, scoring these models on test data, and plotting examples of predictions
    - `info`: Information on Job IDs, checkpoint files
    - `plots`
    - `predictions`
    - `scores`
  - `elm_initial_experimentation`: Contains code for initial experimentation into an Extreme Learning Machine (ELM) based surrogate model.

## Environment Set-Up Instructions
These instructions are relevant to those with a Mac M1 processor.

1. Create virtual environment
```
conda env create -f requirements.yml -n ml-gcm-env
conda activate ml-gcm-env

# If making plots using scripts in `/analysis`, run the following command
brew install graphviz
```

For non-Mac M1 users, replace `tensorflow-macos` and `tensorflow-metal` in `requirements.yml` with the relevant version of TensorFlow, and run above command. 

## Preprocessing data
(Preprocessing is run on CPU partition.)

1. Download the OASIS simulation data and grid to `/data/raw_data`
2. Run the following commands in the remote terminal:
  ```
  # To preprocess data for the LW schema:
  sbatch jobscripts/preprocessing_lw.sh

  # To preprocess data for the SW for dynamical inputs (p, T, rho)
  sbatch jobscripts/preprocessing_sw_dynamical.sh
  ```
3. The output files will be saved to `/data/preprocessed_data/rnn_lw/dynamical` for the LW schema or `/data/preprocessed_data/rnn_sw/dynamical` for the SW dynamical schema

## Training a model
(Training is run on GPU partition.)

1. Change relevant paths in `/model/rnn_sw/vars_sw.py` and `/model/rnn_lw/vars_lw.py`
2. Alter the input hyperparameters in `/jobscripts/rnn_lw_train.sh` and `/jobscripts/rnn_sw_dynamical_train.sh`
3. Run the following commands in the remote terminal:
  ```
  # Train a model for the LW schema
  sbatch jobscripts/run_container.sh rnn_lw_train.sh

  # Train a model for the SW schema for dynamical inputs (p, T, rho)
  sbatch jobscripts/run_container.sh rnn_sw_dynamical_train.sh
  ```

## Evaluating a trained model
(Evaluation requires the GPU partition.)

### Getting model predictions
1. Alter the file `jobscripts/get_model_predictions.sh` with the Job IDs corresponding to trained models of interest.
2. Run the command `sbatch jobscripts/run_container.sh get_model_predictions.sh` to output model predictions to `analysis/trained-models/predictions/` for models corresponding to the specified Job IDs.

### Scoring models & plotting predictions
3. Run the command `sbatch jobscripts/run_container.sh score_models.sh` to output model scores to `analysis/trained-models/info/scores.json` and plots of predictions vs. true for specific samples to `analysis/trained-models/plots`.

## Running predictions using trained models

1. Put raw `.h5` data files in `./data/raw_data`
2. See `example.ipynb` for example on how to use main function in `main.py`.

## Inference instructions
These instructions are for making inference using SW model (.onnx format) in C++.

### Versions
```
Linux compute-0-18.local 3.10.0-1160.49.1.el7.x86_64 #1 SMP Tue Nov 30 15:51:32 UTC 2021 x86_64 x86_64 x86_64 GNU/Linux
Python 3.10.12
cmake version 3.27.2
gcc (GCC) 10.1.0
```

### Onnxruntime installation steps
```
cd ~/oasis-rt-surrogate/inference
git clone --recursive https://github.com/Microsoft/onnxruntime.git
cd onnxruntime
git checkout v1.10.0
./build.sh --config Release  --build_shared_lib --parallel
```

### Compile and run inference
```
# C++
gcc inference.cpp -I ../include/ -o inference -L/~/oasis-rt-surrogate/inference/onnxruntime/build/Linux/Release -lonnxruntime -std=c++11 -lstdc++ -lhdf5
./inference

# C
gcc inference.c -I ../include/ -o inference -L/~/oasis-rt-surrogate/inference/onnxruntime/build/ease -lonnxruntime -lhdf5
./inference
```