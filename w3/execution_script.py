import argparse
import configparser

from train_mlp import main

parser = argparse.ArgumentParser()
parser.add_argument('index', type=int)
args = parser.parse_args()

config = configparser.ConfigParser()
config.read('config.ini')
sections = config.sections()

section = sections[args.index]

arguments = dict(config.items(section))

arguments['units'] = arguments['units'].split(',')
arguments['activation'] = arguments['activation'].split(',')
arguments['metrics'] = arguments['metrics'].split(',')

main(arguments)


