import util
import math
from sklearn.svm import SVC

tuned_params = [[0.01, 100, "rbf"], [0.001, 1.0, "rbf"]]

class SVMClassifier:
    def __init__(self, gamma = "scale", c = 1.0, kernel = "rbf"):
        self.type = "support vector machine"
        self.c = c
        self.gamma = gamma
        self.classifier = SVC(kernel=kernel,C=c, gamma=gamma)

    def train(self, trainingData, trainingLabels):
        self.classifier.fit(trainingData, trainingLabels)

    def classify(self, testData):
        guesses = []
        for datum in testData:
            guesses.append(self.classifier.predict([datum]))
        return guesses

def classify_with_tuned_params(i, trainingData, trainingLabels, testingData):
    classifier = SVMClassifier(tuned_params[i][0], tuned_params[i][1], tuned_params[i][2])
    classifier.train(trainingData, trainingLabels)
    return classifier.classify(testingData)