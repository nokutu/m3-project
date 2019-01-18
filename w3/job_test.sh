#!/bin/bash
#SBATCH --ntasks 4
#SBATCH --mem 16G
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/
#SBATCH --output logs/%x_%u_%j.out

source venv/bin/activate
#python m3-project/w3/train_mlp.py
#python m3-project/w3/mlp_svm.py
python m3-project/w3/mlp_bow.py --model_file /home/grupo06/work/model_2048-1024_relu-relu_categorical_crossentropy_sgd_accuracy_64_16_False_64.h5