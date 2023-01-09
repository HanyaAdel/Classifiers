import util
import math
from sklearn.neural_network import MLPClassifier


class MLP:
    def __init__(self, activation_function = "relu", learning_rate_init = 0.001):
        self.type = "multi layer perceptron"
        self.activation_function = activation_function
        self.learning_rate_init = learning_rate_init
        self.classifier = MLPClassifier(activation= activation_function, learning_rate_init= learning_rate_init, max_iter=1000)

    def train(self, trainingData, trainingLabels):
        self.classifier.fit(trainingData, trainingLabels)

    def classify(self, testData):
        guesses = []
        for datum in testData:
            guesses.append(self.classifier.predict([datum]))
        return guesses