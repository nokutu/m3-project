#!/bin/bash
#SBATCH --job-name group06-w3
#SBATCH --ntasks 4
#SBATCH --mem 4G
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/
#SBATCH --output logs/%x_%u_%j.out
#SBATCH --array=1-30

source venv/bin/activate

python m3-project/w3/execution_script.py ${SLURM_ARRAY_TASK_ID}
