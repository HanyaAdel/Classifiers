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
	for train_row in train:
		dist = util.euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))

	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(k):
		neighbors.append(distances[i][0])
	return neighbors

class KnnClassifier:
	def __init__( self, legalLabels, neighbors):
		self.legalLabels = legalLabels
		self.type = "knn"
		self.k = neighbors
   

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
			train_datum = list(np.append(features[i],self.trainingLabels[i]))
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
			train_datum = list(np.append(features[i],None))
			test_set.append(train_datum)
		self.test_set = test_set


		guesses = []
		
		for test_datum in test_set:
			train_set = self.train_set
			k = self.k
			guess = predict_classification(train_set, test_datum, k)
			guesses.append(guess)
		return guesses
		