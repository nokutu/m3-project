import cv2
import random
from typing import List
from ..model.picture import Picture


class DenseSIFT:

    def __init__(self, step_size: int):
        self._sift = cv2.xfeatures2d.SIFT_create()
        self._step_size = step_size

    def compute(self, filenames: List[str]):
        def _worker(filename):
            if filename not in _images:
                _images[filename] = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

            des = self._compute(_images[filename])

            return des

        return [_worker(filename) for filename in filenames]

    def _compute(self, img):
        kps = []
        for x in range(0, img.shape[1], self._step_size):
            for y in range(0, img.shape[0], self._step_size):
                size = self._step_size * random.uniform(1, 3)
                kp = cv2.KeyPoint(x, y, size)
                kps.append(kp)
        _, des = self._sift.compute(img, kps)
        return Picture(img.shape, kps, des)


_images = dict()  # cache images