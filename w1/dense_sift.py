import cv2
import numpy as np


class DenseSift:

    step_size: int

    def __init__(self, step_size: int):
        self.step_size = step_size

    def dense_sift(self, sift, img: np.ndarray):
        kp = [cv2.KeyPoint(x, y, self.step_size) for y in range(0, img.shape[0], self.step_size)
              for x in range(0, img.shape[1], self.step_size)]
        _, des = sift.compute(img, kp)

        return des
