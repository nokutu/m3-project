from typing import Dict

from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
import keras

from .interface import ModelInterface


class DeepV1Model(ModelInterface):
    model: Model

    def __init__(self):
        self.model = None

    def build(self, input_size: int, n_classes: int, **kwargs) -> Model:
        model = keras.applications.vgg16.VGG16(include_top=True, weights=None, input_tensor=None,
                                               input_shape=(input_size, input_size, 3), pooling='max', classes=8)
        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model

        return model

    def generate_parameters(self, index) -> Dict:
        return {}

    def get_amount_parameters(self) -> int:
        return self.model.count_params()
