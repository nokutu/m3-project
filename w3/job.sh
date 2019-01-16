#!/bin/bash
#SBATCH --ntasks 4
#SBATCH --mem 4G
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/
#SBATCH --output logs/%x_%u_%j.out

source venv/bin/activate
#python m3-project/w3/mlp_MIT_8_scene.py
python m3-project/w3/mlp_model.py