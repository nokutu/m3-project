from typing import Dict

from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD

from .interface import ModelInterface


class DeepModel(ModelInterface):

    model: Model

    def __init__(self):
        self.model = None

    def build(self, input_size: int, n_classes: int, **kwargs) -> Model:
        inputs = Input(shape=(input_size, input_size, 3))

        x = Conv2D(input_size, (3, 3), activation='relu')(inputs)
        x = Conv2D(input_size, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(input_size//2, (3, 3), activation='relu')(x)
        x = Conv2D(input_size//2, (3, 3), activation='relu')(x)
        x = Conv2D(input_size//2, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(input_size//4, (3, 3), activation='relu')(x)
        x = Conv2D(input_size//4, (3, 3), activation='relu')(x)
        x = Conv2D(input_size//4, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Flatten()(x)

        x = Dense(64, activation='relu')(x)

        x = Dropout(0.5)(x)
        predictions = Dense(n_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=predictions)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model

        return model

    def generate_parameters(self, index) -> Dict:
        return {}

    def get_amount_parameters(self) -> int:
        return self.model.count_params()
