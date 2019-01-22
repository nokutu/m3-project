import os
import pickle

from utils import args_to_str, str_to_args


def save_model_weights(args, model, history):
    history_file = os.path.join(args.output_dir, 'history_{}.pkl'.format(args_to_str(args)))
    print('Saving training history to {}...'.format(history_file))
    with open(history_file, 'wb') as pickle_file:
        pickle.dump(history.history, pickle_file)

    model_file = os.path.join(args.output_dir, 'model_{}.h5'.format(args_to_str(args)))
    print('Saving model to {}...'.format(model_file))
    model.save(model_file)

    weights_file = os.path.join(args.output_dir, 'model_{}_weights.h5'.format(args_to_str(args)))
    print('Saving weights to {}...'.format(weights_file))
    model.save_weights(weights_file)

    print('Finished!')


def load_model_from_weights(weights_file):
    args_str = os.path.splitext(os.path.basename(weights_file))[0].split('_', 1)[1]
    args_str = args_str.replace('categorical_crossentropy', 'categorical-crossentropy')
    args = str_to_args(args_str)
    image_size = args.patch_size if args.patches else args.image_size
    # TODO build model

    model.load_weights(weights_file)
    return model
