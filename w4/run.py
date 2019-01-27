import argparse
import os
import pickle

from keras import callbacks

from model import build_model, get_optimizer
from utils import get_train_generator, get_validation_generator, config_to_str, get_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=int)
    parser.add_argument('-d', '--dataset_dir', type=str, default='/home/grupo06/MIT_split')
    parser.add_argument('-o', '--output_dir', type=str, default='/home/grupo06/work')
    parser.add_argument('-l', '--log_dir', type=str, default='/home/grupo06/logs/tensorboard')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=250)
    parser.add_argument('-p', '--patience', type=int, default=5)
    parser.add_argument('-tf', '--train-full', action='store_true', default=False)
    return parser.parse_args()


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

    model = build_model(
        optimizer=config['optimizer'],
        lr=config['learning_rate'],
        decay=config['decay'],
        momentum=config['momentum'],
        loss=config['loss'],
        classes=8,
        use_imagenet=not args.train_full
    )
    # model.summary()

    train_generator = get_train_generator(args.dataset_dir, config['batch_size'])
    validation_generator = get_validation_generator(args.dataset_dir, config['batch_size'])

    tb_callback = callbacks.TensorBoard(log_dir=os.path.join(args.log_dir, config_to_str(config)))
    es_callback = callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=args.patience, verbose=0,
                                          mode='auto', baseline=None, restore_best_weights=True)

    last_epoch = 0

    history = None
    if not args.train_full:
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=config['epochs'],
            verbose=2,
            callbacks=[tb_callback, es_callback],
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            workers=4)

        # this is a small hack: https://github.com/keras-team/keras/issues/1766
        last_epoch = len(history.history['loss'])

        print('\nFine-tuning top layers done. Training full network now...\n')

    for layer in model.layers:
        layer.trainable = True

    optimizer = get_optimizer(
        optimizer=config['optimizer'],
        lr=config['learning_rate'] * config['second_fit_lr_fraction'],
        decay=config['decay'],
        momentum=config['momentum']
    )

    model.compile(optimizer, model.loss, model.metrics)

    history2 = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=config['epochs'],
        verbose=2,
        callbacks=[tb_callback, es_callback],
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        workers=4,
        initial_epoch=last_epoch)

    model_file = os.path.join(args.output_dir, 'nasnet__{}.h5'.format(config_to_str(config)))
    model.save(model_file)
    print('Model saved to {}'.format(model_file))

    if history:
        for k in history.history.keys():
            history.history[k].extend(history2.history[k])
    else:
        history = history2
    history_file = os.path.join(args.output_dir, 'history__{}.pkl'.format(config_to_str(config)))
    with open(history_file, 'wb') as f:
        pickle.dump(history.history, f)
    print('History saved to {}'.format(history_file))

    print('Best accuracy:', max(history.history['val_acc']))


if __name__ == '__main__':
    main()
