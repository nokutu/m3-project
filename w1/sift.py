import cv2
import numpy as np

STEP_SIZE = 32


def sift_method(sift: cv2.xfeatures2d_SIFT, img: np.ndarray):
    _, des = sift.detectAndCompute(img, None)

    return des
