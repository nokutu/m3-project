import argparse

import numpy as np


def get_random_config(args: argparse.Namespace):
    np.random.seed(args.index)

    optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    losses = ['categorical_crossentropy']
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    momenta = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    decays = [0, 0.1, 0.05, 0.01]
    second_fit_lr_fractions = [1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]

    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'optimizer': np.random.choice(optimizers),
        'loss': np.random.choice(losses),
        'learning_rate': np.random.choice(learning_rates),
        'momentum': np.random.choice(momenta),
        'decay': np.random.choice(decays),
        'second_fit_lr_fraction': np.random.choice(second_fit_lr_fractions),
        'index': args.index
    }
    return config


def get_config(args):
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'optimizer': 'Adam',
        'loss': 'categorical_crossentropy',
        'learning_rate': 0.00005,
        'momentum': 0.0,
        'decay': 0.0001,
        'second_fit_lr_fraction': 1.0,
        'index': args.index
    }
    return config


def config_to_str(config):
    s = []
    for k, v in sorted(config.items(), key=lambda t: t[0]):
        s.append('{}={}'.format(k, v))
    return '__'.join(s)


def str_to_config(line):
    d = {}
    for pair in line.split('__'):
        k, v = pair.split('=')
        d[k] = v
    return d
