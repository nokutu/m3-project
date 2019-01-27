from keras import applications, Model
from keras.layers import Dense

from .optimizer import get_optimizer


def build_model(optimizer: str, lr: float, decay: float, momentum: float, loss: str, classes: int,
                freeze=True) -> Model:
    base_model = applications.nasnet.NASNetMobile(
        input_shape=None,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        pooling=None
    )

    for layer in base_model.layers:
        layer.trainable = not freeze

    base_model.layers.pop()
    x = base_model.layers[-1].output  # shape: (None, 1056)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(256, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=x)

    optimizer = get_optimizer(optimizer, lr, decay, momentum)

    model.compile(optimizer, loss, metrics=['accuracy'])

    return model
