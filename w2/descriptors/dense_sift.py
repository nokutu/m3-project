from typing import List
import multiprocessing.dummy as mp

import cv2
import numpy as np

from model.picture import Picture


class DenseSIFT:

    def __init__(self, step_size: int = 2):
        self._sift = cv2.xfeatures2d.SIFT_create()
        self._step_size = step_size
        self._scales = [1, 3, 5, 7, 9]
        self._images = dict()  # cache images

    def compute(self, filenames: List[str]):
        def _worker(filename):
            if filename not in self._images:
                self._images[filename] = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            des = self._compute(self._images[filename])
            return des

        with mp.Pool() as p:
            descriptors = p.map(_worker, filenames)

        return descriptors

    def _compute(self, img: np.ndarray):
        kps = []
        for x in range(0, img.shape[1], self._step_size):
            for y in range(0, img.shape[0], self._step_size):
                for s in self._scales:
                    kp = cv2.KeyPoint(x, y, s)
                    kps.append(kp)
        kps, des = self._sift.compute(img, kps)

        return Picture(img.shape[:2], kps, des)
