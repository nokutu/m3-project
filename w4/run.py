import argparse
import os
import pickle

import numpy as np
from keras import applications
from keras import backend as K
from keras import callbacks
from keras import optimizers
from keras.layers import Dense
from keras.models import Model

from model.TestCallback import TestCallback
from utils import get_train_generator, get_validation_generator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=int)
    parser.add_argument('-d', '--dataset_dir', type=str, default='/home/grupo06/MIT_split')
    parser.add_argument('-o', '--output_dir', type=str, default='/home/grupo06/work')
    parser.add_argument('-l', '--log_dir', type=str, default='/home/grupo06/logs/tensorboard')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-p', '--patience', type=int, default=5)
    parser.add_argument('-x', '--extend', action='store_true', default=False)
    return parser.parse_args()


def get_config(args: argparse.Namespace):
    np.random.seed(args.index)

    optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    losses = ['categorical_crossentropy']
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    momenta = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    lr_fraction_decays = [0, 0.1]

    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'optimizer': np.random.choice(optimizers),
        'loss': np.random.choice(losses),
        'learning_rate': np.random.choice(learning_rates),
        'momentum': np.random.choice(momenta),
        'decay': np.random.choice(lr_fraction_decays)
    }
    return config


def config_to_str(config):
    s = []
    for k, v in sorted(config.items(), key=lambda t: t[0]):
        s.append('{}={}'.format(k, v))
    return '__'.join(s)


def build_model(optimizer_name: str, lr: float, decay_fraction: float, momentum: float, loss: str, classes: int,
                freeze=True) -> Model:
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

    optimizer = optimizers.get(optimizer_name)
    K.set_value(optimizer.lr, lr)
    if hasattr(optimizer, 'momentum'):
        K.set_value(optimizer.momentum, momentum)
    if hasattr(optimizer, 'decay'):
        K.set_value(optimizer.decay, lr * decay_fraction)

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

    model = build_model(optimizer_name=config['optimizer'], lr=config['learning_rate'], decay_fraction=config['decay'],
                        momentum=config['momentum'], loss=config['loss'], classes=8)
    model.summary()

    train_generator = get_train_generator(args.dataset_dir, config['batch_size'])
    validation_generator = get_validation_generator(args.dataset_dir, config['batch_size'])

    # tb_callback = callbacks.TensorBoard(log_dir=os.path.join(args.log_dir, config_to_str(config)))
    es_callback = callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=args.patience, verbose=0,
                                          mode='auto', baseline=None, restore_best_weights=True)
    test_callback = TestCallback()

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=config['epochs'],
        verbose=2,
        callbacks=[es_callback, test_callback],
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        workers=4
    )

    print('Completed first training, initiating full nn retrain')

    for layer in model.layers:
        layer.trainable = True

    model.compile(model.optimizer, model.loss, model.metrics)

    history2 = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=config['epochs'],
        verbose=2,
        callbacks=[test_callback, es_callback],
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        workers=4
    )

    model_file = os.path.join(args.output_dir, 'nasnet__{}.h5'.format(config_to_str(config)))
    model.save(model_file)
    print('Model saved to {}'.format(model_file))

    for key in history.history.keys():
        history.history[key].extend(history2.history[key])

    history_file = os.path.join(args.output_dir, 'history__{}.pkl'.format(config_to_str(config)))
    with open(history_file, 'wb') as f:
        pickle.dump(history.history, f)
    print('History saved to {}'.format(history_file))


if __name__ == '__main__':
    main()
