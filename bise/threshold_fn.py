import numpy as np


def tanh_threshold(x):
    return 1/2 * np.tanh(x) + 1/2
