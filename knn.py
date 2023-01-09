import util
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


tuned_params = [[4,2], [9, 1]]

class KnnClassifier:

    def __init__(self, neighbors = 1, distance_metric = 1):
        self.type = "knn"
        self.knn = KNeighborsClassifier(n_neighbors=neighbors, p = distance_metric)

    def train(self, trainingData, trainingLabels):
        self.trainingData = trainingData
        self.trainingLabels = trainingLabels

        self.knn.fit(trainingData, trainingLabels) # training


    def classify(self, testData):

        guesses = []
        for datum in testData:
            guesses.append(self.knn.predict([datum]))

        return guesses

def classify_with_tuned_params(i, trainingData, trainingLabels, testingData):
    classifier = KnnClassifier(tuned_params[i][0], tuned_params[i][1])
    classifier.train(trainingData, trainingLabels)
    return classifier.classify(testingData)