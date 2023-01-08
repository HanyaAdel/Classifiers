import util
import math
from sklearn.naive_bayes import GaussianNB


class BayesClassifier:
    def __init__(self, v_smoothing):
        self.type = "naivebayes"
        self.probs = {}
        self.extra = False
        self.classifier = GaussianNB(var_smoothing = v_smoothing)

    
    def train(self, trainingData, trainingLabels):
        self.classifier.fit(trainingData, trainingLabels)

    def classify(self, testData):
        guesses = []
        for datum in testData:
            guesses.append(self.classifier.predict([datum]))
        return guesses