#!/bin/bash
#SBATCH --ntasks 4
#SBATCH --mem 4G
#SBATCH --partition mhigh
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/
#SBATCH --output logs/%x_%u_%j.out
#SBATCH --error logs/%x_%u_%j.err

source venv/bin/activate
#python m3-project/w3/mlp_MIT_8_scene.py
python m3-project/w3/patch_based_mlp_MIT_8_scene.py