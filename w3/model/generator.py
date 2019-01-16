from keras_preprocessing.image import ImageDataGenerator


def train_validation_generator(dataset_dir: str, image_size: int, batch_size: int):
    train_datagen, test_datagen = _data_generator()
    # this is a generator that will read pictures found in
    # subfolders of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        dataset_dir + '/train',  # this is the target directory
        target_size=(image_size, image_size),  # all images will be resized to IMG_SIZExIMG_SIZE
        batch_size=batch_size,
        classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        dataset_dir + '/test',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        class_mode='categorical')

    return train_generator, validation_generator


def _data_generator():
    # this is the dataset configuration we will use for training
    # only rescaling
    train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)

    # this is the dataset configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    return train_datagen, test_datagen
