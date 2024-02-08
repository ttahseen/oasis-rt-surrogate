# ML-GCM

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

# Environment Set-Up Instructions
These instructions are relevant to those with a Mac M1 processor.

1. Create virtual environment
```
conda env create -f requirements.yml -n ml-gcm-env
conda activate ml-gcm-env
brew install graphviz
```

For non-Mac M1 users, replace `tensorflow-macos` and `tensorflow-metal` in `requirements.yml` with the relevant version of TensorFlow, and run above command. 

