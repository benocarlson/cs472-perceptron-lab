import numpy as np
from utils import *

BIAS = 1


class Perceptron:

    def __init__(self, num_inputs, init_weights=0):
        self.num_inputs = num_inputs
        self.weights = np.zeros([self.num_inputs + 1], dtype=float)  # Include bias weight
        self.init_weights = init_weights
        self.weights[:] = init_weights
        self.epoch_history = []

    def fire(self, features):
        if len(features) + 1 != len(self.weights):
            return
        total = 0.0
        for i in range(len(self.weights)):
            if i == len(features):
                total += BIAS * self.weights[i]
            else:
                total += features[i] * self.weights[i]
        if total > 0.0:
            return 1
        else:
            return 0

    def predict(self, inputs):
        # expect inputs to be an array of arrays of features
        # output an array of predictions, one for each array of features
        predictions = np.empty([len(inputs)])
        predictions[:] = None
        for i in range(len(inputs)):
            predictions[i] = self.fire(inputs[i])
        return predictions

    def fit(self, inputs, targets, learning_rate, do_shuffle=False, deterministic=False, epochs=10):
        # expect inputs to be an array of arrays of features
        # expect targets to be an array of targets, one per array of features
        best_weights = self.get_weights()
        count = 0
        old_score = 0.0
        total_epochs = 0
        self.epoch_history.append([total_epochs, self.score(inputs, targets)])
        while count < epochs:
            count += 1
            total_epochs += 1
            if do_shuffle:
                epoch_inputs, epoch_targets = shuffle(inputs, targets)
            else:
                epoch_inputs = inputs
                epoch_targets = targets
            for i in range(len(epoch_inputs)):
                output = self.fire(epoch_inputs[i])
                if output != epoch_targets[i]:
                    for j in range(len(self.weights)):
                        if j == len(self.weights) - 1:
                            weight_change = learning_rate * (epoch_targets[i] - output) * BIAS
                        else:
                            weight_change = learning_rate * (epoch_targets[i] - output) * epoch_inputs[i][j]
                        self.weights[j] += weight_change
            score = self.score(inputs, targets)
            if old_score < score:
                if not deterministic:
                    count = 0
                old_score = score
            if score >= old_score:
                best_weights = self.get_weights()
            self.epoch_history.append([total_epochs, score])
        self.weights = best_weights
        return total_epochs

    def score(self, inputs, targets, alt_weights=None):
        # FIXME use alt_weights input for temporary weights
        temp = self.get_weights()
        if alt_weights is not None:
            self.weights = alt_weights
        predictions = self.predict(inputs)
        if predictions is None or len(predictions) != len(targets):
            return
        hits = 0
        for i in range(len(targets)):
            if predictions[i] == targets[i]:
                hits += 1
        self.weights = temp
        return hits / len(targets)

    def get_weights(self):
        return np.copy(self.weights)
