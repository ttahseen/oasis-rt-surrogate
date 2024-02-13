#!/bin/bash

echo $CUDA_VISIBLE_DEVICES

eval "$(/anaconda/bin/conda shell.bash hook)"
conda activate ml-gcm-env
python /home/ucaptp0/oasis-rt-surrogate/analysis/trained-models/get_model_predictions.py 649951 649952 649965 649967 649968 649970
conda deactivate
