from keras import Model
from keras.callbacks import Callback


class RetrainFullNetworkCallback(Callback):
    model: Model

    def __init__(self, model: Model):
        super().__init__()
        self.model = model

    def on_train_end(self, logs=None):
        pass
