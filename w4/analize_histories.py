import os
import pickle


def load_history(output_dir):
    best_acc = 0
    best_history = None
    best_file = None
    for file in os.listdir(output_dir):
        if file.startswith('history'):
            print('Loading training history from {}...'.format(file))
            with open(file, 'r') as pickle_file:
                history = pickle.load(pickle_file)

                if history.history['val_acc'][-1] > best_acc:
                    best_acc = history.history['val_acc'][-1]
                    best_history = history
                    best_file = file
    return best_history, best_file, best_acc
