#!/usr/bin/env bash
#SBATCH --job-name w4
#SBATCH --ntasks 4
#SBATCH --mem 16G
#SBATCH --partition mhigh,mlow
#SBATCH --qos masterlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/
#SBATCH --output logs/%x_%u_%A_%a.out
#SBATCH --array=1-100%3

source venv/bin/activate
python m3-project/w4/train.py ${SLURM_ARRAY_TASK_ID} -o /home/grupo06/work/full_dataset_2 -d /home/mcv/datasets/MIT_split -l /home/grupo06/logs/tensorboard/full_dataset_2
