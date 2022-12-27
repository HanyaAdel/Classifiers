import util
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class KnnClassifier:
    def __init__(self, neighbors, distance_metric):
        self.type = "knn"
        self.k = neighbors
        self.distance_metric = distance_metric
        self.knn = KNeighborsClassifier(n_neighbors=neighbors, p = distance_metric)

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        self.trainingData = trainingData
        self.trainingLabels = trainingLabels
        self.validationData = validationData
        self.validationLabels = validationLabels

        self.knn.fit(trainingData, trainingLabels) # training


    def classify(self, testData):

        guesses = []
        for datum in testData:
            guesses.append(self.knn.predict([datum]))

        return guesses
