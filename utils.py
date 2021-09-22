import numpy as np


def shuffle(data, labels):
    shuffled_data = np.empty(np.shape(data))
    shuffled_labels = np.empty(np.shape(labels))
    ordering = np.arange(len(labels))
    np.random.shuffle(ordering)
    for i in range(len(ordering)):
        shuffled_labels[i] = labels[ordering[i]]
        shuffled_data[i] = data[ordering[i]]
    return shuffled_data, shuffled_labels
