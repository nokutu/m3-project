import os
import argparse

from keras import callbacks

from models.basic_model import basic_model
from load_data import get_train_generator, get_validation_generator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=int)
    parser.add_argument('-d', '--dataset_dir', type=str, default='/home/mcv/datasets/MIT_split')
    parser.add_argument('-o', '--output_dir', type=str, default='/home/grupo06/work/w5')
    parser.add_argument('-l', '--log_dir', type=str, default='/home/grupo06/logs/tensorboard/w5')
    parser.add_argument('-i', '--input_size', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    train_generator = get_train_generator(args.dataset_dir, args.input_size, args.batch_size)
    validation_generator = get_validation_generator(args.dataset_dir, args.input_size, args.batch_size)

    model = basic_model(args.input_size, train_generator.num_classes)
    model.summary()

    early_stopping = callbacks.EarlyStopping(patience=10, verbose=1)
    tensorboard = callbacks.TensorBoard(log_dir=os.path.join(args.log_dir, str(args.index)))

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=args.epochs,
        verbose=2,
        callbacks=[early_stopping, tensorboard],
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        workers=4
    )

    print('Accuracy: {}'.format(max(history.history['val_acc'])))


if __name__ == '__main__':
    main()
