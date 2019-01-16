import os

import numpy as np
from PIL import Image
from sklearn.feature_extraction import image

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


def generate_image_patches_db(in_directory, out_directory, patch_size=64):
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    total = 2688
    count = 0
    for split_dir in os.listdir(in_directory):
        if not os.path.exists(os.path.join(out_directory, split_dir)):
            os.makedirs(os.path.join(out_directory, split_dir))

        for class_dir in os.listdir(os.path.join(in_directory, split_dir)):
            if not os.path.exists(os.path.join(out_directory, split_dir, class_dir)):
                os.makedirs(os.path.join(out_directory, split_dir, class_dir))

            for imname in os.listdir(os.path.join(in_directory, split_dir, class_dir)):
                count += 1
                print('Processed images: ' + str(count) + ' / ' + str(total), end='\r')
                im = Image.open(os.path.join(in_directory, split_dir, class_dir, imname))
                patches = image.extract_patches_2d(np.array(im), (patch_size, patch_size), max_patches=1.0)
                for i, patch in enumerate(patches):
                    patch = Image.fromarray(patch)
                    patch.save(
                        os.path.join(out_directory, split_dir, class_dir, imname.split(',')[0] + '_' + str(i) + '.jpg'))
    print('\n')
