#!/bin/bash
#requesting one node
#SBATCH -N1
#requesting 1 cores
#SBATCH -n1
#SBATCH --mail-user=ucaptp0@ucl.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --mem=350G 

source /share/apps/anaconda3-2021.05/etc/profile.d/conda.sh #for bash 
conda activate
conda activate ml-gcm-env
python /home/ucaptp0/oasis-rt-surrogate/model/rnn_sw/preprocessing_sw_dynamical.py
conda deactivate