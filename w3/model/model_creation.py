from typing import List

from keras.layers import Dense, Reshape
from keras.models import Sequential, Model


def _model_layer(model: Sequential, units: List[int], activation: List[str], name: str) -> Model:
    if len(units) != len(activation):
        raise Exception('ERROR: feature layer parameters length is not equal!')
    for i in range(len(activation)):
        model.add(Dense(units=units[i], activation=activation[i], name=name + str(i)))
    return model


def model_creation(image_size: int, units: List[int], activation: List[str], loss: str, optimizer: str,
                   metrics: List[str], svm=False, test=False) -> Model:
    # Build the Multi Layer Perceptron model
    model = Sequential()
    model.add(Reshape((image_size * image_size * 3,), input_shape=(image_size, image_size, 3), name='data'))
    _model_layer(model, units, activation, 'fc')
    if not svm:
        if not test:
            model.add(Dense(units=8, activation='softmax', name='prob'))
        else:
            model.add(Dense(units=8, activation='linear', name='prob'))
    _model_compile(model, loss, optimizer, metrics)
    return model


def _model_compile(model: Model, loss: str, optimizer: str, metrics: List[str]) -> Model:
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    return model
