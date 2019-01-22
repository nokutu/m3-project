import os
import argparse
import pickle


def args_to_str(args: argparse.Namespace) -> str:
    out = ''
    out += '-'.join(map(str, args.units))
    out += '_' + '-'.join(args.activation)
    out += '_' + args.loss
    out += '_' + args.optimizer
    out += '_' + '-'.join(args.metrics)
    out += '_' + str(args.image_size)
    out += '_' + str(args.batch_size)
    out += '_' + str(args.patches)
    out += '_' + str(args.patch_size)
    return out


def str_to_args(s: str) -> argparse.Namespace:
    args = argparse.Namespace()
    params = s.split('_')

    args.units = list(map(int, params[0].split('-')))
    args.activation = params[1].split('-')
    args.loss = params[2]
    args.optimizer = params[3]
    args.metrics = params[4].split('-')
    args.image_size = int(params[5])
    args.batch_size = int(params[6])
    args.patches = bool(params[7])
    args.patch_size = int(params[8])

    if args.loss == 'categorical-crossentropy':
        args.loss = 'categorical_crossentropy'

    return args


def get_best_model():
    max_acc = 0
    max_f = None

    for f in os.listdir('data/history'):
        with open(os.path.join('data/history', f), 'rb') as file:
            history = pickle.load(file)
            acc = history['val_acc'][-1]
            if acc > max_acc:
                max_acc = acc
                max_f = f

    print('Best model: ', max_f)
    print('Accuracy: ', max_acc)
