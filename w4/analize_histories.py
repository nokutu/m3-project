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

                row = [
                    config.get('batch_size'),
                    config.get('decay'),
                    config.get('epochs'),
                    config.get('learning_rate'),
                    config.get('loss'),
                    config.get('momentum'),
                    config.get('optimizer'),
                    config.get('second_fit_lr_fraction'),
                    history['acc'][best_index],
                    history['loss'][best_index],
                    history['val_acc'][best_index],
                    history['val_loss'][best_index]
                ]
                data.append(row)

    return pandas.DataFrame(data, columns=['batch_size', 'decay', 'epochs', 'learning_rate', 'loss', 'momentum',
                                           'optimizer', 'second_fit_lr_fraction', 'best_train_acc',
                                           'best_train_loss', 'best_val_acc', 'best_val_loss'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default='/home/grupo06/work')

    return parser.parse_args()


def main():
    args = parse_args()
    df = load_dataframe(args.output_dir)
    pandas.set_option('display.max_columns', 500)
    pandas.set_option('display.width', 200)
    print(df[df['best_val_acc'] > 0.90])


if __name__ == '__main__':
    main()
