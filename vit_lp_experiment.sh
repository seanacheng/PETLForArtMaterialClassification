#!/bin/bash
#SBATCH --job-name=vit_lp_experiment
#SBATCH --output=vit_lp_experiment_%j.log
#SBATCH --error=vit_lp_experiment_%j.err
#SBATCH --time=0-03:00
#SBATCH --mem=2000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Activate the micromamba environment
/cluster/tufts/cs152l3dclass/shared/bin/micromamba activate l3d_2024f_cuda_readonly
python --version

# Run the Python script
python vit_lp_experiment.py
