#!/bin/bash
#SBATCH --job-name grupo06-w3
#SBATCH --ntasks 4
#SBATCH --mem 8G
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/
#SBATCH --output logs/%x_%A_%a.out
#SBATCH --array=64-85

source venv/bin/activate
#python m3-project/w3/train_mlp.py
#python m3-project/w3/mlp_bow.py --model_file /home/grupo06/work/model_2048-1024_relu-relu_categorical_crossentropy_sgd_accuracy_64_16_False_64.h5

python m3-project/w3/run.py m3-project/w3/config.ini ${SLURM_ARRAY_TASK_ID}
