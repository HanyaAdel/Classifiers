
import matplotlib.pyplot as plt
from sklearn import tree



class DecisionTreeClassifier():

	def __init__(self, criterion = "gini", max_depth = None, min_samples_split = 2, 
	min_samples_leaf = 1, max_leaf_nodes = None):
		self.type = "decisiontree"
		self.classifier = tree.DecisionTreeClassifier(criterion = criterion, max_depth= max_depth, 
		min_samples_split= min_samples_split, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes)


	def train(self, trainingData, trainingLabels):
	
		# Performing training
		self.classifier.fit(trainingData, trainingLabels)


	
	def test(self, testData):
		guesses = []
		for datum in testData:
			guesses.append(self.classifier.predict([datum]))

		return guesses
	
	def plotTree(self, bestValues, trainingData, trainingLabels):
		self.classifier = tree.DecisionTreeClassifier(criterion = bestValues[0], max_depth= bestValues[1], 
		min_samples_split= bestValues[2], min_samples_leaf=bestValues[3], max_leaf_nodes=bestValues[4])

		self.classifier.fit(trainingData, trainingLabels)
		plt.figure(figsize=(100,70),dpi = 100)
		tree.plot_tree(self.classifier, filled=True, fontsize=10)
		plt.savefig('decision_tree.png')
		plt.show()        
