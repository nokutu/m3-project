from keras import layers
from keras import models
from keras import optimizers


def conv2d_bn(x, filters, kernel_size, padding='same', strides=1):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    return x


def OscarNet(input_size: int, n_classes: int):
    inputs = layers.Input(shape=(input_size, input_size, 3))

    x = conv2d_bn(inputs, 32, 3)
    x = layers.MaxPooling2D(3)(x)

    x = conv2d_bn(x, 64, 3)
    x = conv2d_bn(x, 64, 3)
    x = layers.MaxPooling2D(2)(x)

    x = conv2d_bn(x, 128, 3)
    x = conv2d_bn(x, 128, 3)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    predictions = layers.Dense(n_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=predictions)

    opt = optimizers.Adam()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
