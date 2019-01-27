#!/usr/bin/env bash
#SBATCH --job-name w4
#SBATCH --ntasks 4
#SBATCH --mem 16G
#SBATCH --partition mhigh,mlow
#SBATCH --qos masterlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/
#SBATCH --output logs/%x_%u_%A_%a.out
#SBATCH --array=4,1,44,132,21,80,30,127,24,29,193,170,51,166,93,198,65,122%6

source venv/bin/activate
python m3-project/w4/train.py ${SLURM_ARRAY_TASK_ID} -o /home/grupo06/work/full_dataset -d /home/mcv/datasets/MIT_split