from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras.models import Model


def DeepNet(input_size: int, n_classes: int) -> Model:
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

    return model