from .interface import ModelInterface
from .shallow import ShallowNet
from .deep import DeepNet
from .baseline import BaselineNet
from .oscarnet import OscarNet
from .jorgenet import JorgeNet
from .deep_v1 import DeepV1Model

from .load_data import get_test_generator
from .load_data import get_validation_generator
from .load_data import get_train_generator
