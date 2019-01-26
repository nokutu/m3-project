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
        pooling=None,
        classes=1000)

    for layer in base_model.layers:
        layer.trainable = not freeze

    base_model.layers.pop()
    my_dense = Dense(classes, activation='softmax', name='predictions')
    model = Model(inputs=base_model.input, outputs=my_dense(base_model.layers[-1].output))

    optimizer = get_optimizer(optimizer, lr, decay, momentum)

    model.compile(optimizer, loss, metrics=['accuracy'])

    return model
