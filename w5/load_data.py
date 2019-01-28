import os

from keras_preprocessing.image import ImageDataGenerator, DirectoryIterator


def get_train_generator(dataset_dir: str, input_size: int, batch_size: int) -> DirectoryIterator:
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(dataset_dir, 'train'),
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode='categorical',
        seed=42)

    return train_generator


def get_validation_generator(dataset_dir: str, input_size: int, batch_size: int, shuffle=True) -> DirectoryIterator:
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_generator = test_datagen.flow_from_directory(
        directory=os.path.join(dataset_dir, 'test'),
        target_size=(input_size, input_size),
        batch_size=batch_size,
        shuffle=shuffle,
        class_mode='categorical',
        seed=42)

    return validation_generator

