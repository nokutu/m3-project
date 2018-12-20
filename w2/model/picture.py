from typing import List, Tuple

import cv2
from functional import seq
import numpy as np


class Picture:

    size: Tuple[int, int]
    keypoints: List[Tuple[int, int]]
    descriptors: np.ndarray

    def __init__(self, size: Tuple[int, int], kps: List[cv2.KeyPoint], des: np.ndarray):
        self.size = size
        self.keypoints = seq(kps).map(lambda k: k.pt).to_list()
        self.descriptors = des
