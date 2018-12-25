import numpy as np


def histogram_intersection_kernel(x, u):
    n_samples, n_features = x.shape
    # K = np.zeros(shape=(n_samples, 1), dtype=np.float)
    # for d in range(n_samples):
    #     K[d][0] = np.sum(np.minimum(x[d], u[d]), axis=1)
    return np.sum(np.minimum(x, u), axis=1)
