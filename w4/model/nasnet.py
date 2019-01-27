from keras import applications, Model
from keras.layers import Dense, BatchNormalization, Activation, Dropout

from .optimizer import get_optimizer


def build_model(optimizer: str, lr: float, decay: float, momentum: float, loss: str, classes: int,
                freeze=True, pretrained=True) -> Model:
    if pretrained:
        base_model = applications.nasnet.NASNetMobile(
            input_shape=None,
            include_top=True,
            weights='imagenet',
            input_tensor=None,
            pooling=None,
            classes=1000
        )

        for layer in base_model.layers:
            layer.trainable = not freeze

        base_model.layers.pop()
        x = base_model.layers[-1].output  # shape: (None, 1056)

        """x = Dense(512, name='fc1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(256, name='fc2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)"""

        x = Dense(classes, name='predictions')(x)
        x = BatchNormalization()(x)
        x = Activation('softmax')(x)

        model = Model(inputs=base_model.input, outputs=x)
    else:
        model = applications.nasnet.NASNetMobile(
            input_shape=None,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None,
            classes=classes
        )

    optimizer = get_optimizer(optimizer, lr, decay, momentum)

    model.compile(optimizer, loss, metrics=['accuracy'])

    return model
