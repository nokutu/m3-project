#!/bin/bash
#SBATCH --job-name group06-w3
#SBATCH --ntasks 4
#SBATCH --mem 4G
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/
#SBATCH --output logs/%x_%A_%a.out
#SBATCH --array=0-63

source venv/bin/activate

python m3-project/w3/execution_script.py m3-project/w3/config.ini ${SLURM_ARRAY_TASK_ID}
