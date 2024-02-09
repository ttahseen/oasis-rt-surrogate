#!/bin/bash
#SBATCH -N1
#SBATCH -n1
#SBATCH --mail-user=ucaptp0@ucl.ac.uk
#SBATCH --mail-type=ALL

source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh #for bash 
conda activate
conda activate ml-gcm-env
python /home/ucaptp0/oasis-rt-surrogate/model/rnn_sw/preprocessing_sw_dynamical.py
conda deactivate