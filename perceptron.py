import util
import math
from sklearn.neural_network import MLPClassifier

tuned_params = [[(200, 100, 50), "relu", 0.001],[(50, 25, 12), "identity", 0.1]]
class MLP:
    def __init__(self, hidden_layer_sizes=(100,), activation_function = "relu", learning_rate_init = 0.001):
        self.type = "multi layer perceptron"
        self.activation_function = activation_function
        self.learning_rate_init = learning_rate_init
        self.classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,activation= activation_function, learning_rate_init= learning_rate_init, max_iter=1000)

    def train(self, trainingData, trainingLabels):
        self.classifier.fit(trainingData, trainingLabels)

    def classify(self, testData):
        guesses = []
        for datum in testData:
            guesses.append(self.classifier.predict([datum]))
        return guesses

def classify_with_tuned_params(i, trainingData, trainingLabels, testingData):
    classifier = MLP(tuned_params[i][0], tuned_params[i][1], tuned_params[i][2])
    classifier.train(trainingData, trainingLabels)
    return classifier.classify(testingData)