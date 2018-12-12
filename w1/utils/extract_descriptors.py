import cv2
import multiprocessing.dummy as mp


def extract_descriptors(sift, filenames):
    def _worker(filename):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        kps, des = sift.detectAndCompute(img, None)
        return des

    with mp.Pool(processes=4) as p:
        descriptors = p.map(_worker, filenames)

    return descriptors
