from Perceptron import *
from arff import Arff


debug_mat = Arff("./linsep2nonorigin.arff")
debug_data = debug_mat.data[:, 0:-1]
debug_labels = debug_mat.data[:, -1].reshape(-1, 1)
debug_perceptron = Perceptron(debug_mat.features_count, init_weights=0)

debug_perceptron.fit(debug_data, debug_labels, learning_rate=0.1, do_shuffle=False, deterministic=True, epochs=10)
debug_score = debug_perceptron.score(debug_data, debug_labels)
print("DEBUG")
print("Accuracy = [{:.2f}]".format(debug_score))
print("Final Weights =", debug_perceptron.get_weights())
print('\n')


eval_mat = Arff("./data_banknote_authentication.arff")
data = eval_mat.data[:, 0:-1]
labels = eval_mat.data[:, -1].reshape(-1, 1)
eval_perceptron = Perceptron(eval_mat.features_count, init_weights=0)


eval_perceptron.fit(data, labels, learning_rate=0.1, do_shuffle=False, deterministic=True, epochs=10)
eval_score = eval_perceptron.score(data, labels)
print("EVALUATION")
print("Accuracy = [{:.2f}]".format(eval_score))
print("Final Weights =", eval_perceptron.get_weights())


