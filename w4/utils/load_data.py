import os

from keras_preprocessing.image import ImageDataGenerator


def load_dataset(path):
    filenames, labels = [], []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        if not os.path.isdir(label_path):
            continue
        for image in os.listdir(label_path):
            image_path = os.path.join(label_path, image)
            if not image_path.endswith('.jpg'):
                continue
            filenames.append(image_path)
            labels.append(label)
    return filenames, labels


def get_train_generator(dataset_dir: str, batch_size: int):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(dataset_dir, 'train'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
    )

    return train_generator


def get_validation_generator(dataset_dir: str, batch_size: int, shuffle=True):
    test_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    validation_generator = test_datagen.flow_from_directory(
        directory=os.path.join(dataset_dir, 'test'),
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=shuffle,
        class_mode='categorical')

    return validation_generator
