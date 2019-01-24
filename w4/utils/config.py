import argparse

import numpy as np


def get_config(args: argparse.Namespace):
    np.random.seed(args.index)

    optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    losses = ['categorical_crossentropy']
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    momenta = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    decays = [0, 0.1, 0.05, 0.01]
    second_fit_lr_fractions = [1, 0.1, 0.01]

    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'optimizer': np.random.choice(optimizers),
        'loss': np.random.choice(losses),
        'learning_rate': np.random.choice(learning_rates),
        'momentum': np.random.choice(momenta),
        'decay': np.random.choice(decays),
        'second_fit_lr_fraction': np.random.choice(second_fit_lr_fractions)
    }
    return config
