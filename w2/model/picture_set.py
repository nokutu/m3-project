import numpy as np
from typing import List
from joblib import hash
from model.picture import Picture


class PictureSet:
    pictures: List[Picture]

    def __init__(self, pictures: List[Picture]):
        self.pictures = pictures

    def __repr__(self):
        return "PictureSet(pictures=" + hash(np.vstack([p.descriptors for p in self.pictures])) + ')'
