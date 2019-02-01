from typing import Dict

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, BatchNormalization, LeakyReLU
from keras.layers import Input
from keras.models import Model

from .interface import ModelInterface


class DeepV1Model(ModelInterface):
    model: Model

    def __init__(self):
        self.model = None

    def build(self, input_size: int, n_classes: int, **kwargs) -> Model:
        inputs = Input(shape=(input_size, input_size, 3))

        x = Conv2D(input_size // 2, (3, 3))(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(input_size // 2, (3, 3))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(input_size // 4, (3, 3))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(input_size // 4, (3, 3))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(input_size // 4, (3, 3))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(input_size // 8, (3, 3))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(input_size // 8, (3, 3))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(input_size // 8, (3, 3))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Flatten()(x)

        # x = Dropout(0.5)(x)
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        predictions = Dense(n_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=predictions)

        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model

        return model

    def generate_parameters(self, index) -> Dict:
        return {}

