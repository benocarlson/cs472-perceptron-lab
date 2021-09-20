import numpy as np


def shuffle(array):
    shuffled = np.copy(array)
    np.random.shuffle(shuffled)
    return shuffled
