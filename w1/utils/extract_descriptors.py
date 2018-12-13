from typing import List
import multiprocessing.dummy as mp

import cv2
import random


class DescriptorExtractor:

    def __init__(self, n_features=0):
        self._sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features)
        self._images = dict()  # cache images

    def _compute(self, img):
        pass

    def compute(self, filenames: List[str]):
        def _worker(filename):
            if filename not in self._images:
                self._images[filename] = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            des = self._compute(self._images[filename])
            return des

        with mp.Pool(processes=4) as p:
            descriptors = p.map(_worker, filenames)

        return descriptors


class SIFT(DescriptorExtractor):

    def __init__(self, n_features: int):
        super().__init__(n_features)

    def _compute(self, img):
        _, des = self._sift.detectAndCompute(img, None)
        return des


class DenseSIFT(DescriptorExtractor):

    def __init__(self, step_size: int):
        super().__init__()
        self._step_size = step_size

    def _compute(self, img):
        kps = []
        for x in range(0, img.shape[1], self._step_size):
            for y in range(0, img.shape[0], self._step_size):
                size = self._step_size * random.uniform(1, 3)
                kp = cv2.KeyPoint(x, y, size)
                kps.append(kp)
        _, des = self._sift.compute(img, kps)
        return des
