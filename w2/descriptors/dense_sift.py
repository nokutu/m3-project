import os
from typing import List
import multiprocessing.dummy as mp
import pickle

import cv2
import numpy as np

from model.picture import Picture


class DenseSIFT:

    def __init__(self, step_size: int = 2, memory: str = None):
        self.step_size = step_size
        self.scales = [1, 3, 5, 7, 9]
        self.memory = memory

        self._sift = cv2.xfeatures2d.SIFT_create()
        self._images = dict()  # cache images

    def compute(self, filenames: List[str]):
        # Try to load descriptors from cache
        descriptors = self._load(filenames)
        if descriptors is not None:
            return descriptors

        def _worker(filename):
            if filename not in self._images:
                self._images[filename] = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            des = self._compute(self._images[filename])
            return des

        with mp.Pool() as p:
            descriptors = p.map(_worker, filenames)

        # Save descriptors to cache
        self._save(descriptors, filenames)

        return descriptors

    def _compute(self, img: np.ndarray):
        kps = []
        for x in range(0, img.shape[1], self.step_size):
            for y in range(0, img.shape[0], self.step_size):
                for s in self.scales:
                    kp = cv2.KeyPoint(x, y, s)
                    kps.append(kp)
        kps, des = self._sift.compute(img, kps)

        return Picture(img.shape[:2], kps, des)

    def _save(self, descriptors, filenames):
        if self.memory:
            cache_file = self._cache_file(filenames)
            os.makedirs(self.memory, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(descriptors, f)

    def _load(self, filenames):
        if self.memory:
            cache_file = self._cache_file(filenames)
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    descriptors = pickle.load(f)
                return descriptors

    def _cache_file(self, filenames):
        h = format(hash((filenames, self.step_size)), 'x')
        return os.path.join(self.memory, '{}.pkl'.format(h))
