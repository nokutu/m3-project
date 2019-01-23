from keras.callbacks import Callback
from keras import backend as K


class TestCallback(Callback):

    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        print('Learning rate:', K.get_value(self.model.optimizer.lr))
        if hasattr(self.model.optimizer, 'decay'):
            print('Decay:', K.get_value(self.model.optimizer.decay))
        print('Iterations:', K.get_value(self.model.optimizer.iterations))
