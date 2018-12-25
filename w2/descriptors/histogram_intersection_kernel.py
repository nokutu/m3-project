import numba as numba
import numpy as np


@numba.njit()
def histogram_intersection_kernel(x, u):
    n_samples, n_features = x.shape
    K = np.zeros((x.shape[0], u.shape[0]), dtype=np.float32)
    for d in numba.prange(n_samples):
        K[d, :] = np.sum(np.minimum(x[d], u), axis=1)
    return K
