import os
import glob
import random
import multiprocessing.dummy as mp

import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator, img_to_array
from keras.applications import imagenet_utils

SEED = 42
VALIDATION_SPLIT = 0.0


def load_image(image_path, input_size):
    img = Image.open(image_path)
    img = img.resize((input_size, input_size), resample=Image.BICUBIC)
    img = img_to_array(img)
    label = image_path.split(os.path.sep)[-2]
    return img, label


def preprocess_input(x):
    return imagenet_utils.preprocess_input(x, mode='caffe')


def get_train_validation_generator(dataset_dir: str, input_size: int, batch_size: int) -> (NumpyArrayIterator, NumpyArrayIterator):
    image_paths = glob.glob(os.path.join(dataset_dir, 'train', '*/*'))
    random.seed(SEED)
    random.shuffle(image_paths)

    with mp.Pool() as p:
        x, y = zip(*p.starmap(load_image, [(image_path, input_size) for image_path in image_paths]))
    x = np.array(x)
    y = np.array(y)

    lb = LabelBinarizer()
    y = lb.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=VALIDATION_SPLIT, random_state=SEED)

    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    train_generator = train_datagen.flow(x_train, y_train, batch_size, shuffle=False)

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow(x_test, y_test, batch_size, shuffle=False)

    print('Found {} images.'.format(len(image_paths)))
    return train_generator, validation_generator


def get_test_generator(dataset_dir: str, input_size: int, batch_size: int) -> NumpyArrayIterator:
    image_paths = glob.glob(os.path.join(dataset_dir, 'test', '*/*'))

    with mp.Pool() as p:
        x, y = zip(*p.starmap(load_image, [(image_path, input_size) for image_path in image_paths]))
    x = np.array(x)
    y = np.array(y)

    lb = LabelBinarizer()
    y = lb.fit_transform(y)

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow(x, y, batch_size, shuffle=False)

    print('Found {} images.'.format(len(image_paths)))
    return test_generator
