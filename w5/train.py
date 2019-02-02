import argparse
import os

import numpy as np
from keras import callbacks

from model import OscarNet
from utils.load_data import get_train_validation_generator, get_test_generator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=int, help='Seed to generate the parameters')
    parser.add_argument('-d', '--dataset_dir', type=str, default='/home/mcv/datasets/MIT_split')
    parser.add_argument('-o', '--output_dir', type=str, default='/home/grupo06/work/w5')
    parser.add_argument('-l', '--log_dir', type=str, default='/home/grupo06/logs/tensorboard/w5')
    parser.add_argument('-i', '--input_size', type=int, default=128)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    train_generator, _ = get_train_validation_generator(args.dataset_dir, args.input_size, args.batch_size)
    test_generator = get_test_generator(args.dataset_dir, args.input_size, args.batch_size)

    num_classes = len(np.unique(train_generator.y, axis=0))
    model = OscarNet(args.input_size, num_classes)
    model.summary()

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    tensorboard = callbacks.TensorBoard(log_dir=os.path.join(args.log_dir, str(args.index)))

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=args.epochs,
        verbose=2,
        callbacks=[reduce_lr, early_stopping, tensorboard],
        validation_data=test_generator,
        validation_steps=test_generator.n // test_generator.batch_size,
        workers=4
    )

    test_metrics = model.evaluate_generator(
        test_generator,
        steps=(test_generator.n // test_generator.batch_size) + 1,
        verbose=1,
        workers=4
    )

    for metric, value in zip(model.metrics_names, test_metrics):
        print('{}: {}'.format(metric, value))


if __name__ == '__main__':
    main()
