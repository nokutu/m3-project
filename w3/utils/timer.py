import time


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        print('{}...'.format(self.name))
        return self

    def __exit__(self, *args):
        self.end = time.time()
        print('{}: {:.6f}s'.format(self.name, self.end - self.start))
