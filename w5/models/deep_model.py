from typing import Dict

from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD

from .model_interface import ModelInterface


class DeepModel(ModelInterface):

    model: Model

    def __init__(self):
        self.model = None

    def build(self, input_size: int, n_classes: int, **kwargs) -> Model:
        inputs = Input(shape=(input_size, input_size, 3))

        x = Conv2D(32, (3, 3))(inputs)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 3))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(16, (3, 3))(x)
        x = Activation('relu')(x)
        x = Conv2D(16, (3, 3))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(8, (3, 3))(x)
        x = Conv2D(8, (3, 3))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Flatten()(x)

        x = Dropout(0.5)(x)
        x = Dense(64)(x)
        x = Activation('relu')(x)

        x = Dropout(0.5)(x)
        x = Dense(n_classes)(x)
        predictions = Activation('softmax')(x)

        model = Model(inputs=inputs, outputs=predictions)

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model

        return model

    def generate_parameters(self, index) -> Dict:
        return {}

    def get_amount_parameters(self) -> int:
        return self.model.count_params()
