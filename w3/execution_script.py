import configparser

config = configparser.ConfigParser()
config.read('config.ini')
sections = config.sections()

for section in sections:

    batch_size = (sections[i], 'batch_size')
    image_size = (section, 'image_size')
    units = (section, 'units')
    activation = (section, 'activation')
    loss = (section, 'loss')
    optimizer = (section, 'optimizer')
    metrics = (section, 'metrics')
    patch = config.get_boolean(section, 'patch')
    patch_size
