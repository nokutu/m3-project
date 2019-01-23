from keras import optimizers
from keras import backend as K
from keras.optimizers import Optimizer


def get_optimizer(optimizer, lr, decay, momentum) -> Optimizer:
    optimizer = optimizers.get(optimizer)
    K.set_value(optimizer.lr, lr)
    if hasattr(optimizer, 'momentum'):
        K.set_value(optimizer.momentum, momentum)
    if hasattr(optimizer, 'decay'):
        K.set_value(optimizer.decay, decay)
    return optimizer
