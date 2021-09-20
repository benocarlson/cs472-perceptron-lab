import numpy as np
from utils import *


class Perceptron:

    def __init__(self, num_inputs, learning_rate=0.1, min_change=0.0):
        self.num_inputs = num_inputs
        self.weights = np.zeros([self.num_inputs], dtype=float)
        self.learning_rate = learning_rate
        self.min_change = min_change

    def fire(self, features):
        if len(features) != len(self.weights):
            return
        total = 0.0
        for i in range(len(self.weights)):
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

    def fit(self, inputs, targets):
        # expect inputs to be an array of arrays of features
        # expect targets to be an array of targets, one per array of features
        best_weights = self.get_weights()
        count = 0
        old_score = 0.0
        while count < 10:
            count += 1
            shuffled = shuffle(inputs)
            for i in range(len(shuffled)):
                output = self.fire(shuffled[i])
                if output != targets[i]:
                    for j in range(len(self.weights)):
                        weight_change = self.learning_rate * (targets[i] - output) * shuffled[i][j]
                        self.weights[i] += weight_change
            score = self.score(inputs, targets)
            if old_score < score:
                count = 0
                old_score = score
                best_weights = self.get_weights()
        self.weights = best_weights
        return old_score

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
