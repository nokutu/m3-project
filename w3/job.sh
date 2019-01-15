#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH --mem 4000
#SBATCH -D /home/grupo06/ # working directory
#SBATCH -p mhigh,mhigh # or mlow Partition to submit to
#SBATCH --gres gpu:1 # Para pedir Pascales MAX 8
#SBATCH -o logs/%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e logs/%x_%u_%j.err # File to which STDERR will be written

cd m3-project/w3
source ../venv/bin/activate
python mlp_MIT_8_scene.py
