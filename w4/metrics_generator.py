import argparse
import numpy as np
import pickle

from keras.engine.saving import load_model

from utils import get_validation_generator
from utils.metrics import save_accuracy, save_loss, save_confusion_matrix
from utils.utils import str_to_config


def get_model_history(model_file):
    model = load_model(model_file, compile=False)
    history_file = model_file.replace('nasnet', 'history').replace('.h5', '.pkl')
    with open(history_file, 'rb') as file:
        history = pickle.load(file)

    return model, history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('-o', '--output_dir', type=str, default='/home/grupo06/work')
    parser.add_argument('-d', '--dataset_dir', type=str, default='/home/mcv/datasets/MIT_split')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    model, history = get_model_history(args.model_file)
    print('Maximum accuracy: ', max(history['val_acc']))

    config_str = args.model_file.split('/')[-1].replace('nasnet__', '').replace('.h5', '')
    print(config_str)
    config = str_to_config(config_str)

    model.compile(config['optimizer'], config['loss'], metrics=['accuracy'])

    save_accuracy(history, args.model_file.replace('nasnet', 'accuracy').replace('.h5', '.png'))
    save_loss(history, args.model_file.replace('nasnet', 'loss').replace('.h5', '.png'))

    validation_generator = get_validation_generator(args.dataset_dir, args.batch_size, shuffle=False)
    y_pred = model.predict_generator(
        validation_generator,
        steps=(validation_generator.samples // validation_generator.batch_size) + 1
    )
    y_pred = np.argmax(y_pred, axis=1)
    save_confusion_matrix(validation_generator.classes, y_pred, validation_generator.class_indices,
                          args.model_file.replace('nasnet', 'confusion').replace('.h5', '.png'))


if __name__ == '__main__':
    main()
