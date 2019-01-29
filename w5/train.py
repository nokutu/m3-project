import argparse
import os
import pickle
from typing import Dict

import pandas as pd
from keras import callbacks

from model import ModelInterface, BasicModel, DeepModel
from model.load_data import get_train_generator, get_validation_generator, get_test_generator

PATIENCE = 10

model_map: Dict[str, ModelInterface] = {
    'basic': BasicModel(),
    'deep': DeepModel()
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=model_map.keys(), help='Model to use')
    parser.add_argument('index', type=int, help='Seed to generate the parameters')
    parser.add_argument('-d', '--dataset_dir', type=str, default='/home/mcv/datasets/MIT_split')
    parser.add_argument('-o', '--output_dir', type=str, default='/home/grupo06/work/w5')
    parser.add_argument('-l', '--log_dir', type=str, default='/home/grupo06/logs/tensorboard/w5')
    parser.add_argument('-i', '--input_size', type=int, default=64)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    train_generator = get_train_generator(args.dataset_dir, args.input_size, args.batch_size)
    validation_generator = get_validation_generator(args.dataset_dir, args.input_size, args.batch_size)
    test_generator = get_test_generator(args.dataset_dir, args.input_size, args.batch_size)

    model_class = model_map[args.model]
    params = model_class.generate_parameters(args.index)
    model = model_class.build(args.input_size, train_generator.num_classes, **params)
    model.summary()

    early_stopping = callbacks.EarlyStopping(patience=PATIENCE, verbose=1)
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

    test_metrics = model.evaluate_generator(
        test_generator,
        steps=(test_generator.samples // test_generator.batch_size) + 1
    )

    index_names = ['model', 'index']
    indices = [[args.model, args.index]]
    headers = (['amount_parameters', 'params', 'train_acc', 'train_loss', 'val_acc', 'val_loss'] +
               list(map(lambda s: 'test_' + s, model.metrics_names)))
    data = [[model_class.get_amount_parameters(), params, history.history['acc'][-PATIENCE],
             history.history['loss'][-PATIENCE], history.history['val_acc'][-PATIENCE],
             history.history['val_loss'][-PATIENCE]] + test_metrics]

    results = pd.DataFrame(data, columns=headers, index=pd.MultiIndex.from_tuples(indices, names=index_names))

    save_files = True
    result_file = os.path.join(args.output_dir, 'result_{}_{}.pkl'.format(args.model, args.index))

    if os.path.exists(result_file):
        old_results: pd.DataFrame = pd.read_pickle(result_file)
        if results.index[0] in old_results.index:
            old_row = old_results.loc[results.index[0]]
            new_row = results.loc[results.index[0]]

            if new_row.val_acc < old_row.val_acc:
                save_files = False
                print('Combination already calculated. Current is worse than previous. Not replacing.')
            else:
                print('Combination already calculated. Current is better than previous. Replacing.')

    if save_files:
        print('Saving results to {}...'.format(result_file))
        results.to_pickle(result_file)

        history_file = os.path.join(args.output_dir, 'history_{}_{}.pkl'.format(args.model, args.index))
        print('Saving training history to {}...'.format(history_file))
        with open(history_file, 'wb') as pickle_file:
            pickle.dump(history.history, pickle_file)

        model_file = os.path.join(args.output_dir, 'model_{}_{}.h5'.format(args.model, args.index))
        print('Saving model to {}...'.format(model_file))
        model.save(model_file)

    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.width', 200)

    print(results)


if __name__ == '__main__':
    main()
