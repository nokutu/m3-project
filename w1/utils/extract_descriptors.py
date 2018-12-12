from typing import List

import cv2
import multiprocessing.dummy as mp

images = dict()


def extract_descriptors(method, sift: cv2.xfeatures2d_SIFT, filenames: List[str]):
    def _worker(filename):
        if filename not in images:
            images[filename] = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        return method(sift, images[filename])

    with mp.Pool(processes=4) as p:
        descriptors = p.map(_worker, filenames)

    return descriptors
