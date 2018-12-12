import cv2
import numpy as np

STEP_SIZE = 32


def dense_sift(sift, img: np.ndarray):
    kp = [cv2.KeyPoint(x, y, STEP_SIZE) for y in range(0, img.shape[0], STEP_SIZE)
          for x in range(0, img.shape[1], STEP_SIZE)]
    _, descriptors = sift.compute(img, kp)

    return descriptors
