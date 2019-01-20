import argparse
import configparser

from train_mlp import main

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('index', type=int)
parser.add_argument('-d', '--dataset_dir', type=str)
parser.add_argument('-o', '--output_dir', type=str)
parser.add_argument('-pd', '--patches_dir', type=str)
parser.add_argument('-b', '--batch_size', type=int, default=16)
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config)
sections = config.sections()

print('Selecting config {}/{}'.format(args.index, len(sections)))

section = sections[args.index]

arguments = dict(config.items(section))

if 'units' in arguments:
    arguments['units'] = list(map(int, arguments['units'].split(',')))
if 'activation' in arguments:
    arguments['activation'] = arguments['activation'].split(',')
if 'metrics' in arguments:
    arguments['metrics'] = arguments['metrics'].split(',')
if 'patches' in arguments:
    arguments['patches'] = bool(arguments['patches'])

if args.dataset_dir:
    arguments['dataset_dir'] = args.dataset_dir
if args.output_dir:
    arguments['output_dir'] = args.output_dir
if args.patches_dir:
    arguments['patches_dir'] = args.patches_dir
if args.batch_size:
    arguments['batch_size'] = args.batch_size

print(arguments)
main(arguments)
