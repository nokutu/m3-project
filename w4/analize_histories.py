import argparse
import os
import pickle


def load_best_history(output_dir):
    best_acc = 0
    best_history = None
    best_file = None
    for file in os.listdir(output_dir):
        if file.startswith('history'):
            print('Loading training history from {}...'.format(file))
            with open(file, 'rb') as pickle_file:
                history = pickle.load(pickle_file)

                if history.history['val_acc'][-1] > best_acc:
                    best_acc = history.history['val_acc'][-1]
                    best_history = history
                    best_file = file
    return best_history, best_file, best_acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default='/home/grupo06/work')

    return parser.parse_args()


def main():
    args = parse_args()
    best_history, best_file, best_acc = load_best_history(args.output_dir)
    print('Best model: ', best_file)
    print('Accuracy: ', best_acc)


if __name__ == '__main__':
    main()
