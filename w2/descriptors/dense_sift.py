import os
from typing import List, Tuple
import multiprocessing.dummy as mp
import hashlib
import pickle

import cv2
import numpy as np

from model.picture import Picture


class DenseSIFT:

    def __init__(self, step_size: int = 10, scales: Tuple[int] = (4, 8, 12, 16), memory: str = None):
        self.step_size = step_size
        self.scales = scales
        self.memory = memory

        self._sift = cv2.xfeatures2d.SIFT_create()
        self._images = dict()  # cache images

        #print('{}: {}'.format(self.__class__.__name__, vars(self)))

    def compute(self, filenames: List[str]) -> List[Picture]:
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

    def _compute(self, img: np.ndarray) -> Picture:
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
                # https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
                pickle.dump(descriptors, f)

    def _load(self, filenames):
        if self.memory:
            cache_file = self._cache_file(filenames)
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    descriptors = pickle.load(f)
                return descriptors

    def _cache_file(self, filenames):
        h = hashlib.md5()
        h.update(str(self.step_size).encode('utf-8'))
        for scale in self.scales:
            h.update(str(scale).encode('utf-8'))
        for filename in filenames:
            h.update(filename.encode('utf-8'))
        return os.path.join(self.memory, '{}.pkl'.format(h.hexdigest()))
