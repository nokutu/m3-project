from keras import layers
from keras import regularizers
from keras import models
from keras import optimizers


def conv2d_bn(filters, kernel_size, padding='same', strides=1):
    def f(x):
        x = layers.Conv2D(filters=filters, kernel_size=kernel_size,
                          strides=strides, padding=padding,
                          kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)
        return x

    return f


def OscarNet(input_size: int, n_classes: int, initial_filters=32, repetitions=(2, 2, 2)):
    inputs = layers.Input(shape=(input_size, input_size, 3))

    x = conv2d_bn(initial_filters, kernel_size=3)(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)

    filters = initial_filters
    for r in repetitions:
        for i in range(r):
            x = conv2d_bn(filters, kernel_size=3)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        filters *= 2

    x = layers.Dropout(0.5)(x)
    x = conv2d_bn(n_classes, kernel_size=1, padding='valid')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(inputs, x)

    opt = optimizers.Adam()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
