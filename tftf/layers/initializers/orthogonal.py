import numpy as np
import tensorflow as tf


def orthogonal(shape, scale=1., name=None, rng=None, type='float32'):
    if rng is None:
        rng = np.random

    rndn = rng.normal(0., 1., shape).astype(type)
    u, _, v = np.linalg.svd(rndn, full_matrices=False)
    if u.shape == shape:
        return scale * u
    else:
        return scale * v
