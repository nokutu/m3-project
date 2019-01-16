import argparse
import os

from PIL import Image
from colored import stylize, fg
import numpy as np
from sklearn.feature_extraction import image

from model import model_creation
from utils import str_to_args
from utils.softmax import softmax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('-d', '--dataset', type=str, default='/home/mcv/datasets/MIT_split')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args2 = str_to_args(args.model_file.split('/')[-1].split['_'][1:])
    if args2.patch:
        model = model_creation(args2.patch_size, args2.units, args2.activation, args2.loss, args2.optimizer,
                               args2.metrics, test=True)
    else:
        model = model_creation(args2.image_size, args2.units, args2.activation, args2.loss, args2.optimizer,
                               args2.metrics, test=True)
    print(model.summary())

    print(stylize('Done!\n', fg('blue')))
    print(stylize('Loading weights from ' + args.model_file + ' ...\n', fg('blue')))
    print('\n')

    model.load_weights(args.model_file)

    print(stylize('Done!\n', fg('blue')))

    print(stylize('Start evaluation ...\n', fg('blue')))

    directory = args2.dataset + '/test'
    classes = {'coast': 0, 'forest': 1, 'highway': 2, 'inside_city': 3, 'mountain': 4, 'Opencountry': 5,
               'street': 6, 'tallbuilding': 7}
    correct = 0.
    total = 807
    count = 0

    for class_dir in os.listdir(directory):
        cls = classes[class_dir]
        for imname in os.listdir(os.path.join(directory, class_dir)):
            im = Image.open(os.path.join(directory, class_dir, imname))
            if args2.patch:
                patches = image.extract_patches_2d(np.array(im), (args2.patch_size, args2.patch_size), max_patches=1.0)
                out = model.predict(patches / 255.)
            else:
                out = model.predict(im / 255.)
            predicted_cls = np.argmax(softmax(np.mean(out, axis=0)))
            if predicted_cls == cls:
                correct += 1
            count += 1
            print('Evaluated images: ' + str(count) + ' / ' + str(total), end='\r')

    print(stylize('Done!\n', fg('blue')))
    print(stylize('Test Acc. = ' + str(correct / total) + '\n', fg('green')))
