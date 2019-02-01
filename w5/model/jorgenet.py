from typing import Dict

from keras import layers, Model, backend
from keras import models
from keras import optimizers

from model import ModelInterface


class JorgeNet(ModelInterface):

    def generate_parameters(self, index) -> Dict:
        return {}

    def _conv2d_bn(self, x, filters, kernel_size, padding='same', strides=1):
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)
        return x

    def build(self, input_size: int, n_classes: int, **kwargs) -> Model:
        inputs = layers.Input(shape=(input_size, input_size, 3))

        x = self._conv2d_bn(inputs, 64, 3)
        x = layers.MaxPooling2D(3)(x)

        a = self._conv2d_bn(x, 64, 5)
        b = self._conv2d_bn(x, 64, 3)
        c = self._conv2d_bn(x, 64, 1)

        x = layers.concatenate([a, b, c])
        x = layers.MaxPooling2D(3)(x)

        a = self._conv2d_bn(x, 128, 5)
        b = self._conv2d_bn(x, 128, 3)
        c = self._conv2d_bn(x, 128, 1)

        x = layers.concatenate([a, b, c])
        x = layers.MaxPooling2D(3)(x)

        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dropout(0.5)(x)

        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)

        predictions = layers.Dense(n_classes, activation='softmax')(x)

        model = models.Model(inputs=inputs, outputs=predictions)

        opt = optimizers.Adam()
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        return model
