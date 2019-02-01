import abc
from typing import Dict

from keras import Model


# noinspection PyTypeChecker
class ModelInterface:

    @abc.abstractmethod
    def build(self, input_size: int, n_classes: int, **kwargs) -> Model:
        return

    @abc.abstractmethod
    def generate_parameters(self, index) -> Dict:
        return
