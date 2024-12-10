#!/bin/bash
#SBATCH --job-name=vit_ft_experiment
#SBATCH --output=vit_ft_experiment_%j.log
#SBATCH --error=vit_ft_experiment_%j.err
#SBATCH --time=0-01:00
#SBATCH --mem=2000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

# Activate the micromamba environment
/cluster/tufts/cs152l3dclass/shared/bin/micromamba activate l3d_2024f_cuda_readonly
python --version

# Run the Python script
python vit_ft_experiment.py
