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


def rand_split_70_30(data, labels):
    set_size = len(data)
    partition_70 = int(set_size / (10/7))
    shuffled_data, shuffled_labels = shuffle(data, labels)
    data_70 = shuffled_data[:partition_70]
    data_30 = shuffled_data[partition_70:]
    labels_70 = shuffled_labels[:partition_70]
    labels_30 = shuffled_labels[partition_70:]
    return data_70, labels_70, data_30, labels_30
