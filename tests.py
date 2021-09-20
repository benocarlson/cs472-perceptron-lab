from utils import *


def test_shuffle():
    array = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    shuffled = shuffle(array)
    print(str(array))
    print(str(shuffled))

    array_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    array_shuffled = shuffle(array_array)
    print(str(array_array))
    print(str(array_shuffled))

test_shuffle()