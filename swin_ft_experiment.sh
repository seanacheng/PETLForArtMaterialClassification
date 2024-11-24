#!/bin/bash
#SBATCH --job-name=swin_ft_experiment
#SBATCH --output=swin_ft_experiment_%j.log
#SBATCH --error=swin_ft_experiment_%j.err
#SBATCH --time=0-02:00
#SBATCH --mem=2000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Activate the micromamba environment
micromamba activate l3d_2024f_cuda_readonly

# Specify Python version
module load python/3.10.4

# Run the Python script
python swin_ft_experiment.py
