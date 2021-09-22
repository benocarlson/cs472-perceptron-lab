from Perceptron import *
from arff import Arff
import matplotlib.pyplot as plt

lin_sep = Arff("./myLinSep.arff")
lin_sep_data = lin_sep.data[:, 0:-1]
lin_sep_labels = lin_sep.data[:, -1].reshape(-1, 1)


non_lin_sep = Arff("./myNonLinSep.arff")
non_lin_sep_data = non_lin_sep.data[:, 0:-1]
non_lin_sep_labels = non_lin_sep.data[:, -1].reshape(-1, 1)


learning_rates = [0.1, 0.2, 0.5, 1.0]

def run_lin_sep():
    for i in range(len(learning_rates)):
        perceptron = Perceptron(lin_sep.features_count, init_weights=0)
        epochs = perceptron.fit(lin_sep_data, lin_sep_labels, learning_rates[i], do_shuffle=True, deterministic=False, epochs=10)
        final_score = perceptron.score(lin_sep_data, lin_sep_labels)
        print("With 8 linearly separable data")
        print("At a learning rate of ", learning_rates[i])
        print("Arrived at accuracy: ", final_score)
        print("After epochs: ", epochs)
        print("With weights: ", perceptron.get_weights())
        print("\n")


def run_non_lin_sep():
    for i in range(len(learning_rates)):
        perceptron = Perceptron(non_lin_sep.features_count, init_weights=0)
        epochs = perceptron.fit(non_lin_sep_data, non_lin_sep_labels, learning_rates[i], do_shuffle=True, deterministic=False, epochs=10)
        final_score = perceptron.score(non_lin_sep_data, non_lin_sep_labels)
        print("With 8 non-linearly separable data")
        print("At a learning rate of ", learning_rates[i])
        print("Arrived at accuracy: ", final_score)
        print("After epochs: ", epochs)
        print("With weights: ", perceptron.get_weights())
        print("\n")


def plot_lin_sep():
    perceptron = Perceptron(lin_sep.features_count, init_weights=0)
    epochs = perceptron.fit(lin_sep_data, lin_sep_labels, learning_rate=0.1, do_shuffle=False, deterministic=False, epochs=10)
    weights = perceptron.get_weights()
    for i in range(len(lin_sep_data)):
        plt.plot(lin_sep_data[i, 0], lin_sep_data[i, 1], 'bo' if lin_sep_labels[i] == 1 else 'rs')
    x = np.linspace(-1, 1, 10)
    y = (-(weights[0] / weights[1]) * x) + (- weights[2] / weights[1])
    plt.plot(x, y, '-g', label='decision boundary')
    plt.axis([-1, 1, -1, 1])
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.legend(loc='upper left')
    plt.title('Linearly Separable Example')
    plt.show()

def plot_non_lin_sep():
    perceptron = Perceptron(non_lin_sep.features_count, init_weights=0)
    epochs = perceptron.fit(non_lin_sep_data, non_lin_sep_labels, learning_rate=0.1, do_shuffle=False, deterministic=False,
                            epochs=10)
    weights = perceptron.get_weights()
    for i in range(len(non_lin_sep_data)):
        plt.plot(non_lin_sep_data[i, 0], non_lin_sep_data[i, 1], 'bo' if non_lin_sep_labels[i] == 1 else 'rs')
    x = np.linspace(-1, 1, 10)
    y = (-(weights[0] / weights[1]) * x) + (- weights[2] / weights[1])
    plt.plot(x, y, '-g', label='decision boundary')
    plt.axis([-1, 1, -1, 1])
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.legend(loc='upper right')
    plt.title('Non-Linearly Separable Example')
    plt.show()

plot_lin_sep()
plot_non_lin_sep()