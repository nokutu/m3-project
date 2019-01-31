import os

from keras_preprocessing.image import ImageDataGenerator, DirectoryIterator

SEED = 42
VALIDATION_SPLIT = 0


def get_train_generator(dataset_dir: str, input_size: int, batch_size: int) -> DirectoryIterator:
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        validation_split=VALIDATION_SPLIT)

    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(dataset_dir, 'train'),
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode='categorical',
        seed=SEED,
        subset='training',
        interpolation='lanczos')

    return train_generator


def get_validation_generator(dataset_dir: str, input_size: int, batch_size: int) -> DirectoryIterator:
    validation_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=VALIDATION_SPLIT)

    validation_generator = validation_datagen.flow_from_directory(
        directory=os.path.join(dataset_dir, 'train'),
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode='categorical',
        seed=SEED,
        subset='validation',
        interpolation='lanczos')

    return validation_generator


def get_test_generator(dataset_dir: str, input_size: int, batch_size: int) -> DirectoryIterator:
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        directory=os.path.join(dataset_dir, 'test'),
        target_size=(input_size, input_size),
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical',
        interpolation='lanczos')

    return test_generator
