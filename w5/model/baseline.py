from keras import applications
from keras import layers
from keras import models


def BaselineNet(input_size: int, n_classes: int):
    base_model = applications.mobilenet_v2.MobileNetV2(input_shape=(input_size, input_size, 3), include_top=False)

    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    predictions = layers.Dense(n_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])

    return model
