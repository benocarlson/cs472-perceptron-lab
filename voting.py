from Perceptron import *
from arff import Arff
from utils import *
from matplotlib import pyplot as plt
import pandas as pd

mat = Arff("./voting.arff")
data = mat.data[:, 0:-1]
labels = mat.data[:, -1].reshape(-1, 1)


results = np.empty([5, 3])
weights = np.empty([5, mat.features_count + 1])
epoch_histories = []

for i in range(5):
    training_data, training_labels, test_data, test_labels = rand_split_70_30(data, labels)
    perceptron = Perceptron(mat.features_count, init_weights=0)
    epochs = perceptron.fit(training_data, training_labels, learning_rate=0.1, do_shuffle=True, epochs=5)
    training_score = perceptron.score(training_data, training_labels)
    test_score = perceptron.score(test_data, test_labels)
    results[i] = [epochs, training_score, test_score]
    weights[i] = perceptron.get_weights()
    epoch_histories.append(perceptron.epoch_history)

result_dict = {'Iteration': [1, 2, 3, 4, 5], 'Total Epochs': results[:, 0], 'Training Accuracy': results[:, 1], 'Test Accuracy': results[:, 2]}
print(pd.DataFrame(data=result_dict))

epoch_averages = []
i = 0
done = False
while not done:
    epoch_total = 0
    epoch_count = 0
    for j in range(5):
        if i < len(epoch_histories[j]):
            epoch_total += epoch_histories[j][i][1]
            epoch_count += 1
    if epoch_count > 0:
        epoch_averages.append(1 - (epoch_total / epoch_count))
    else:
        done = True
    i += 1
plt.plot(epoch_averages)
plt.axis([0, len(epoch_averages), 0, 0.5])
plt.xlabel('Epoch')
plt.ylabel('Average Misclassification Rate')
plt.title('Average Misclassification Across Epochs')
plt.show()

average_weights = np.empty([mat.features_count + 1])
for j in range(len(average_weights)):
    weight_sum = 0
    for k in range(len(weights)):
        weight_sum += weights[k][j]
    average_weights[j] = weight_sum / len(weights)
w_labels = mat.attr_names
w_labels[-1] = 'BIAS'
w_x = np.arange(len(average_weights))
plt.bar(w_x, average_weights, align='edge')
plt.axis([0, len(average_weights), -1, 1])
plt.ylabel('Average Weight')
plt.xlabel('Feature')
plt.title('Average Weights Across Features')
plt.show()