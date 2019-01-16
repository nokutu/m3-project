import os
from typing import List

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model


def _model_layer(model: Sequential, units: List[int], activation: List[str], name: str) -> Model:
    if len(units) != len(activation):
        raise Exception('ERROR: feature layer parameters length is not equal!')
    for i in range(len(activation)):
        model.add(Dense(units=units[i], activation=activation[i], name=name + str(i)))
    return model


def model_creation(image_size: int, units: List[int], activation: List[str], loss: str, optimizer: str,
                   metrics: List[str]) -> Model:
    # Build the Multi Layer Perceptron model
    model = Sequential()
    model.add(Reshape((image_size * image_size * 3,), input_shape=(image_size, image_size, 3), name='data'))
    _model_layer(model, units, activation, 'fc')
    model.add(Dense(units=8, activation='softmax', name='prob'))
    _model_compile(model, loss, optimizer, metrics)
    return model


def _model_compile(model: Model, loss: str, optimizer: str, metrics: List[str]) -> Model:
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    return model
