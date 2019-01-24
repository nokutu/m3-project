import argparse
import os
import pickle

import pandas

from utils.utils import str_to_config


def load_dataframe(output_dir):
    data = []
    for file in os.listdir(output_dir):
        if file.startswith('history'):
            print('Loading training history from {}...'.format(file))
            config_str = file.replace('history__', '').replace('.pkl', '')
            config = str_to_config(config_str)

            with open(os.path.join(output_dir, file), 'rb') as pickle_file:
                history = pickle.load(pickle_file)
                best_index = history['val_acc'].index(max(history['val_acc']))
                index = config.get('index')
                fit = config.get('second_fit_lr_fraction')
                row = [
                    int(index) if index else index,
                    int(config.get('batch_size')),
                    float(config.get('decay')),
                    int(config.get('epochs')),
                    float(config.get('learning_rate')),
                    config.get('loss'),
                    float(config.get('momentum')),
                    config.get('optimizer'),
                    float(fit) if fit else fit,
                    len(history['val_acc']),
                    history['acc'][best_index],
                    history['loss'][best_index],
                    history['val_acc'][best_index],
                    history['val_loss'][best_index]
                ]
                data.append(row)

    return pandas.DataFrame(data,
                            columns=['index', 'batch_size', 'decay', 'max_epochs', 'learning_rate', 'loss', 'momentum',
                                     'optimizer', 'second_fit_lr_fraction', 'epochs', 'best_train_acc',
                                     'best_train_loss', 'best_val_acc', 'best_val_loss'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default='/home/grupo06/work')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    df = load_dataframe(args.output_dir)
    pandas.set_option('display.max_columns', 500)
    pandas.set_option('display.max_rows', 500)
    pandas.set_option('display.width', 200)
    print(df[(df['best_val_acc'] > 0.90) & (df['second_fit_lr_fraction'].isnull())])
    print(df[(df['best_val_acc'] > 0.90) & (df['second_fit_lr_fraction'].notnull())])
