import argparse
import pickle

from keras.engine.saving import load_model

from utils.metrics import save_accuracy, save_loss, save_confusion_matrix


def get_model_history(model_file):
    with open(model_file, 'rb') as file:
        model = load_model(file)

    history_file = model_file.replace('model', 'history').replace('.h5', '.pkl')
    with open(history_file, 'rb') as file:
        history = pickle.load(file)

    return model, history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_file', type=str)
    parser.add_argument('-o', '--output_dir', type=str, default='/home/grupo06/work')
    return parser.parse_args()


def main():
    args = parse_args()
    model, history = get_model_history(args.model_file)
    print('Maximum accuracy: ', history.history['val_acc'][-1])
    save_accuracy(history, args.model_file.replace('model', 'accuracy').replace('.h5', '.png'))
    save_loss(history, args.model_file.replace('model', 'loss').replace('.h5', '.png'))


if __name__ == '__main__':
    main()
