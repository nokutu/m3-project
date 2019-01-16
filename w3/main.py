import argparse
from model import model_creation
from utils import args_to_str, str_to_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='/home/mcv/datasets/MIT_split')
    parser.add_argument('-u', '--units', type=int, nargs='+', default=[2048, 1024])
    parser.add_argument('-a', '--activation', type=str, nargs='+', default=['relu', 'relu'])
    parser.add_argument('-l', '--loss', type=str, default='categorical_crossentropy')
    parser.add_argument('-o', '--optimizer', type=str, default='sgd')
    parser.add_argument('-m', '--metrics', type=str, nargs='+', default=['accuracy'])
    parser.add_argument('-s', '--image-size', type=int, default=64)
    parser.add_argument('-n', '--names', type=str, nargs='+')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model_creation(args.image_size, args.units, args.activation, args.loss, args.optimizer, args.metrics)
