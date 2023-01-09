import util
import math
from sklearn.svm import SVC

class SVMClassifier:
    def __init__(self, gamma = "scale", c = 1.0):
        self.type = "support vector machine"
        self.c = c
        self.gamma = gamma
        self.classifier = SVC(C=c, gamma=gamma)

    def train(self, trainingData, trainingLabels):
        self.classifier.fit(trainingData, trainingLabels)

    def classify(self, testData):
        guesses = []
        for datum in testData:
            guesses.append(self.classifier.predict([datum]))
        return guesses