import argparse

import keras
import numpy as np
from keras.layers import Dropout
from keras.utils import print_summary, plot_model

parser = argparse.ArgumentParser()
parser.add_argument('index', type=int)
parser.add_argument('-d', '--dataset_dir', type=str)
parser.add_argument('-o', '--output_dir', type=str)
parser.add_argument('-pd', '--patches_dir', type=str)
parser.add_argument('-b', '--batch_size', type=int, default=16)
parser.add_argument('-e', '--extend', action='store_true')
args = parser.parse_args()

np.random.seed(args.index)

optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
learn_rates = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
momenta = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

if args.extend:
    pass

optimizer = np.random.choice(optimizers)
learn_rate = np.random.choice(learn_rates)
momentum = np.random.choice(momenta)

model = keras.applications.nasnet.NASNetMobile(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
print_summary(model)
plot_model(model, to_file='file.png', show_shapes=True, show_layer_names=True)