#!/bin/bash
#SBATCH --ntasks 4
#SBATCH --mem 16G
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/
#SBATCH --output logs/%x_%u_%j.out

source venv/bin/activate
#python m3-project/w3/train_mlp.py --batch_size 256 --patches
#python m3-project/w3/test_mlp.py work/model_4096-2048_relu-relu_categorical_crossentropy_sgd_accuracy_64_256_True_64_weights.h5 --batch_size 256 --patches
#python m3-project/w3/mlp_svm.py
python m3-project/w3/mlp_bow.py work/model_4096-2048_relu-relu_categorical_crossentropy_sgd_accuracy_64_256_True_64.h5