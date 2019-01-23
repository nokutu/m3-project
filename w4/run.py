import os
import argparse
import pickle

import numpy as np
from keras import applications
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.models import Model
from keras import optimizers
from keras import callbacks
from keras import backend as K


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=int)
    parser.add_argument('-d', '--dataset_dir', type=str, default='/home/mcv/datasets/MIT_split')
    parser.add_argument('-o', '--output_dir', type=str, default='/home/grupo06/work')
    parser.add_argument('-l', '--log_dir', type=str, default='/home/grupo06/logs/tensorboard')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-x', '--extend', action='store_true', default=False)
    return parser.parse_args()


def get_config(args: argparse.Namespace):
    np.random.seed(args.index)

    optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    losses = ['categorical_crossentropy']
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    momenta = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

    if args.extend:
        pass

    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'optimizer': np.random.choice(optimizers),
        'loss': np.random.choice(losses),
        'learning_rate': np.random.choice(learning_rates),
        'momentum': np.random.choice(momenta)
    }
    return config


def config_to_str(config):
    s = []
    for k, v in sorted(config.items(), key=lambda t: t[0]):
        s.append('{}={}'.format(k, v))
    return '__'.join(s)


def get_train_generator(dataset_dir: str, batch_size: int):
    train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(dataset_dir, 'train'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator


def get_validation_generator(dataset_dir: str, batch_size: int):
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_generator = test_datagen.flow_from_directory(
        directory=os.path.join(dataset_dir, 'test'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    return validation_generator


def build_model(optimizer: str, lr: float, loss: str, classes: int, freeze=True):
    base_model = applications.nasnet.NASNetMobile(
        input_shape=None,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        pooling=None,
        classes=1000)

    for layer in base_model.layers:
        layer.trainable = not freeze

    base_model.layers.pop()
    my_dense = Dense(classes, activation='softmax', name='predictions')
    model = Model(inputs=base_model.input, outputs=my_dense(base_model.layers[-1].output))

    optimizer = optimizers.get(optimizer)
    K.set_value(optimizer.lr, lr)

    model.compile(optimizer, loss, metrics=['accuracy'])
    return model


def print_setup(config: dict):
    print('\n\tExperimental setup')
    print('\t------------------\n')
    for k, v in sorted(config.items(), key=lambda t: t[0]):
        print('\t{}: {}'.format(k, v))
    print('')


def main():
    args = parse_args()
    config = get_config(args)
    print_setup(config)

    model = build_model(optimizer=config['optimizer'], lr=config['learning_rate'], loss=config['loss'], classes=8)
    #model.summary()

    train_generator = get_train_generator(args.dataset_dir, config['batch_size'])
    validation_generator = get_validation_generator(args.dataset_dir, config['batch_size'])

    tb_callback = callbacks.TensorBoard(log_dir=os.path.join(args.log_dir, config_to_str(config)))

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=config['epochs'],
        verbose=2,
        # callbacks=[tb_callback],
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        workers=4
    )

    model_file = os.path.join(args.output_dir, 'nasnet__{}.h5'.format(config_to_str(config)))
    model.save(model_file)
    print('Model saved to {}'.format(model_file))

    history_file = os.path.join(args.output_dir, 'history__{}.pkl'.format(config_to_str(config)))
    with open(history_file, 'wb') as f:
        pickle.dump(history.history, f)
    print('History saved to {}'.format(history_file))


if __name__ == '__main__':
    main()
