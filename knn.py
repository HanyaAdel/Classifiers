import util
import numpy as np


# Make a classification prediction with neighbors
def predict_classification(train, test_row, k, distance_metric):
    neighbors = get_neighbors(train, test_row, k, distance_metric)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# Locate the most similar neighbors
def get_neighbors(train, test_row, k, distance_metric):
    distances = list()
    if distance_metric == 0:  # 0 is eucildean distance
        for train_row in train:
            dist = util.euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
    else:
        for train_row in train:
            dist = util.manhattanDistance(test_row, train_row)
            distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors


class KnnClassifier:
    def __init__(self, legalLabels, neighbors, distance_metric):
        self.legalLabels = legalLabels
        self.type = "knn"
        self.k = neighbors
        self.distance_metric = distance_metric

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        self.trainingData = trainingData
        self.trainingLabels = trainingLabels
        self.validationData = validationData
        self.validationLabels = validationLabels

        self.size = len(list(trainingData))
        features = [];
        for datum in trainingData:
            feature = list(datum.values())
            features.append(feature)

        train_set = [];
        for i in range(self.size):
            train_datum = list(np.append(features[i], self.trainingLabels[i]))
            train_set.append(train_datum)
        self.train_set = train_set

    def classify(self, testData):
        self.size = len(list(testData))
        features = [];
        for datum in testData:
            feature = list(datum.values())
            features.append(feature)

        test_set = [];
        for i in range(self.size):
            train_datum = list(np.append(features[i], None))
            test_set.append(train_datum)
        self.test_set = test_set

        guesses = []

        for test_datum in test_set:
            train_set = self.train_set
            k = self.k
            distance_metric=self.distance_metric
            guess = predict_classification(train_set, test_datum, k, distance_metric)
            guesses.append(guess)
        return guesses