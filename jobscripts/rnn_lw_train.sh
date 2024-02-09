#!/bin/bash

echo $CUDA_VISIBLE_DEVICES

eval "$(/anaconda/bin/conda shell.bash hook)"
conda activate ml-gcm-env
python /home/ucaptp0/oasis-rt-surrogate/model/rnn_lw/rnn_lw.py "1.0" "0" "32"
conda deactivate