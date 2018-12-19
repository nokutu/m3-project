
class Picture:

    def __init__(self, size: (int, int), kps, des):
        self.size = size
        self.keypoints = kps
        self.descriptors = des