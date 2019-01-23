import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from sklearn.feature_extraction import image

from keras_preprocessing.image import ImageDataGenerator
from multiprocessing import Pool


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


def get_train_generator(path: str, image_size: int, batch_size: int):
    train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(path, 'train'),
        target_size=(image_size, image_size),
        batch_size=batch_size,
        classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        class_mode='categorical')

    return train_generator


def get_validation_generator(path: str, image_size: int, batch_size: int):
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_generator = test_datagen.flow_from_directory(
        directory=os.path.join(path, 'test'),
        target_size=(image_size, image_size),
        batch_size=batch_size,
        classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        class_mode='categorical')

    return validation_generator

