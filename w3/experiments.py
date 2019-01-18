import os
import pickle


def get_best():
    max_acc = 0
    max_f = None

    for f in os.listdir('data/history'):
        with open(os.path.join('data/history', f), 'rb') as file:
            history = pickle.load(file)
            acc = history['val_acc'][-1]
            if acc > max_acc:
                max_acc = acc
                max_f = f

    print('Best file: ', max_f)
    print('Accuracy: ', max_acc)


if __name__ == '__main__':
    get_best()
