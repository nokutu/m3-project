from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Model
from keras.optimizers import SGD


def basic_model(input_size, n_classes):
    inputs = Input(shape=(input_size, input_size, 3))

    x = Conv2D(32, (3, 3))(inputs)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes)(x)

    predictions = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
