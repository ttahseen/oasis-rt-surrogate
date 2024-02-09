#!/bin/bash
#SBATCH -p GPU
#requesting one node
#SBATCH -N1
#requesting 1 V100 GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=ucaptp0@ucl.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --mem=100G

jobscript=$1
echo "Jobscript: $jobscript"
chmod +x /home/ucaptp0/oasis-rt-surrogate/jobscripts/$jobscript

srun singularity exec --nv /home/ucaptp0/ml-gcm-cobweb/singularity-images/ubuntu20_anaconda_cuda11.2.sif /home/ucaptp0/oasis-rt-surrogate/jobscripts/$jobscript