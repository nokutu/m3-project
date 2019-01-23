import argparse
import pickle

from keras.engine.saving import load_model

from utils import get_validation_generator
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
    parser.add_argument('-d', '--dataset_dir', type=str, default='/home/mcv/datasets/MIT_split')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    model, history = get_model_history(args.model_file)
    print('Maximum accuracy: ', history.history['val_acc'][-1])
    save_accuracy(history, args.model_file.replace('model', 'accuracy').replace('.h5', '.png'))
    save_loss(history, args.model_file.replace('model', 'loss').replace('.h5', '.png'))

    validation_generator = get_validation_generator(args.dataset_dir, args.batch_size, False)
    y_pred = model.predict(validation_generator)
    save_confusion_matrix(validation_generator.classes, y_pred,
                          args.model_file.replace('model', 'confusion').replace('.h5', '.png'))


if __name__ == '__main__':
    main()
