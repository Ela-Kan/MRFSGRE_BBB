# Numba helper functions
import numpy as np
from numba import njit


@njit
def sincos(x):
    return np.sin(x), np.cos(x)