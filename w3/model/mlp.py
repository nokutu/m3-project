import os
from typing import List

from keras.layers import Dense, Reshape
from keras.models import Sequential, Model

from utils import str_to_args


def build_model(image_size: int, units: List[int], activation: List[str], optimizer: str, loss: str,
                metrics: List[str], svm=False, test=False) -> Model:
    """Build the Multi Layer Perceptron model."""
    model = Sequential()
    model.add(Reshape((image_size * image_size * 3,), input_shape=(image_size, image_size, 3), name='data'))
    _add_layers(model, units, activation, 'fc')
    if not svm:
        if not test:
            model.add(Dense(units=8, activation='softmax', name='prob'))
        else:
            model.add(Dense(units=8, activation='linear', name='prob'))
    model.compile(optimizer, loss, metrics)
    return model


def _add_layers(model: Sequential, units: List[int], activation: List[str], name: str):
    if len(units) != len(activation):
        raise Exception('ERROR: feature layer parameters length is not equal!')

    for i in range(len(activation)):
        model.add(Dense(units=units[i], activation=activation[i], name=name + str(i+1)))


def load_model_from_weights(weights_file):
    args_str = os.path.splitext(os.path.basename(weights_file))[0].split('_', 1)[1]
    args_str = args_str.replace('categorical_crossentropy', 'categorical-crossentropy')
    args = str_to_args(args_str)
    image_size = args.patch_size if args.patches else args.image_size
    model = build_model(image_size, args.units, args.activation, args.optimizer, args.loss, args.metrics, test=True)
    model.load_weights(weights_file)
    return model
