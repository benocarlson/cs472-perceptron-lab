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


def test_rand_split():
    data = [[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]]
    targets = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
    train_d, train_l, test_d, test_l = rand_split_70_30(data, targets)
    print("Training Data: ", train_d, train_l)
    print("Test Data: ", test_d, test_l)

test_rand_split()