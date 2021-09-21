from Perceptron import *
from arff import Arff

mat = Arff("./data_banknote_authentication.arff")
data = mat.data[:, 0:-1]
labels = mat.data[:, -1].reshape(-1, 1)
perceptron = Perceptron(mat.features_count, init_weights=0)


perceptron.fit(data, labels, learning_rate=0.1, do_shuffle=False, deterministic=True, epochs=10)
score = perceptron.score(data, labels, )
print("Accuracy = [{:.2f}]".format(score))
print("Final Weights =", perceptron.get_weights())


