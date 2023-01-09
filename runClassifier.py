from knn import KnnClassifier
import knn, decision_tree, svm, perceptron, bayes
from bayes import BayesClassifier
from decision_tree import DecisionTreeClassifier
from perceptron import MLP
from svm import SVMClassifier
from matplotlib import pyplot as plt
from processing import calcAccuracy, prepareData

def run_classifier_with_tuned_parameters(faceTrainingData, faceTrainingLabels,\
	faceTestingData, faceTestingLabels,\
	digitTrainingData, digitTrainingLabels,\
	digitTestingData, digitTestingLabels):

	'''BAYES WITH TUNED PARAMETERS ON DIGITS'''
	print("Running bayes on the digits dataset with tuned parameters...")
	guesses = bayes.classify_with_tuned_params(0, trainingData=digitTrainingData, trainingLabels=digitTrainingLabels,
	testingData=digitTestingData)
	calcAccuracy(guesses=guesses, correctLabels=digitTestingLabels)
	print("-------------------------------------------\n")


	'''BAYES WITH TUNED PARAMETERS ON FACES'''
	print("Running Bayes on the faces dataset with tuned parameters...")
	guesses = bayes.classify_with_tuned_params(1, trainingData=faceTrainingData, trainingLabels=faceTrainingLabels,
	testingData=faceTestingData)
	calcAccuracy(guesses=guesses, correctLabels=faceTestingLabels)	
	print("-------------------------------------------\n")

	'''MLP WITH TUNED PARAMETERS ON DIGITS'''
	print("Running MLP on the digits dataset with tuned parameters...")
	guesses = perceptron.classify_with_tuned_params(0, trainingData=digitTrainingData, trainingLabels=digitTrainingLabels,
	testingData=digitTestingData)
	calcAccuracy(guesses=guesses, correctLabels=digitTestingLabels)
	print("-------------------------------------------\n")


	'''MLP WITH TUNED PARAMETERS ON FACES'''
	print("Running MLP on the faces dataset with tuned parameters...")
	guesses = perceptron.classify_with_tuned_params(1, trainingData=faceTrainingData, trainingLabels=faceTrainingLabels,
	testingData=faceTestingData)
	calcAccuracy(guesses=guesses, correctLabels=faceTestingLabels)	
	print("-------------------------------------------\n")


	'''DECISION TREE WITH TUNED PARAMETERS ON DIGITS'''
	print("Running decision tree on the digits dataset with tuned parameters...")
	guesses = decision_tree.classify_with_tuned_params(0, trainingData=digitTrainingData, trainingLabels=digitTrainingLabels,
	testingData=digitTestingData)
	calcAccuracy(guesses=guesses, correctLabels=digitTestingLabels)	
	print("-------------------------------------------\n")


	'''DECISION TREE WITH TUNED PARAMETERS ON FACES'''
	print("Running decision tree on the faces dataset with tuned parameters...")
	guesses = decision_tree.classify_with_tuned_params(1, trainingData=faceTrainingData, trainingLabels=faceTrainingLabels,
	testingData=faceTestingData)
	calcAccuracy(guesses=guesses, correctLabels=faceTestingLabels)
	print("-------------------------------------------\n")

	'''SVM WITH TUNED PARAMETERS ON DIGITS'''
	print("Running SVM on the digits dataset with tuned parameters...")
	guesses = svm.classify_with_tuned_params(0, trainingData=digitTrainingData, trainingLabels=digitTrainingLabels,
	testingData=digitTestingData)
	calcAccuracy(guesses=guesses, correctLabels=digitTestingLabels)	
	print("-------------------------------------------\n")


	'''SVM WITH TUNED PARAMETERS ON FACES'''
	print("Running SVM on the faces dataset with tuned parameters...")
	guesses = svm.classify_with_tuned_params(1, trainingData=faceTrainingData, trainingLabels=faceTrainingLabels,
	testingData=faceTestingData)
	calcAccuracy(guesses=guesses, correctLabels=faceTestingLabels)
	print("-------------------------------------------\n")

	'''KNN WITH TUNED PARAMETERS ON DIGITS'''
	print("Running KNN on the digits dataset with tuned parameters...")
	guesses = knn.classify_with_tuned_params(0, trainingData=digitTrainingData, trainingLabels=digitTrainingLabels,
	testingData=digitTestingData)
	calcAccuracy(guesses=guesses, correctLabels=digitTestingLabels)
	print("-------------------------------------------\n")


	'''KNN WITH TUNED PARAMETERS ON FACES'''
	print("Running KNN on the faces dataset with tuned parameters...")
	guesses = knn.classify_with_tuned_params(1, trainingData=faceTrainingData, trainingLabels=faceTrainingLabels,
	testingData=faceTestingData)
	calcAccuracy(guesses=guesses, correctLabels=faceTestingLabels)	


def runClassifier(faceTrainingData, faceTrainingLabels,\
	faceValidationData, faceValidationLabels, \
	faceTestingData, faceTestingLabels,\
	digitTrainingData, digitTrainingLabels, \
	digitValidationData, digitValidationLabels, \
	digitTestingData, digitTestingLabels):

	
	# runMLP(0, trainingData=digitTrainingData, validationData=digitValidationData, testingData=digitTestingData,
	# trainingLabels=digitTrainingLabels, validationLabels=digitValidationLabels,
	# testingLabels=digitTestingLabels)

	# runMLP(1, trainingData=faceTrainingData, validationData=faceValidationData, testingData=faceTestingData,
	# trainingLabels=faceTrainingLabels, validationLabels=faceValidationLabels,
	# testingLabels=faceTestingLabels)	

	# runSVM(0, trainingData=digitTrainingData, validationData=digitValidationData, testingData=digitTestingData,
	# trainingLabels=digitTrainingLabels, validationLabels=digitValidationLabels,
	# testingLabels=digitTestingLabels)

	# runSVM(1, trainingData=faceTrainingData, validationData=faceValidationData, testingData=faceTestingData,
	# trainingLabels=faceTrainingLabels, validationLabels=faceValidationLabels,
	# testingLabels=faceTestingLabels)		

	# runKNN(0, trainingData=digitTrainingData, validationData=digitValidationData, testingData=digitTestingData,
	# trainingLabels=digitTrainingLabels, validationLabels=digitValidationLabels,
	# testingLabels=digitTestingLabels)
	
	# runKNN(1, trainingData=faceTrainingData, validationData=faceValidationData, testingData=faceTestingData,
	# trainingLabels=faceTrainingLabels, validationLabels=faceValidationLabels,
	# testingLabels=faceTestingLabels)    

	runBayes(0, trainingData=digitTrainingData, validationData=digitValidationData, testingData=digitTestingData,
	trainingLabels=digitTrainingLabels, validationLabels=digitValidationLabels,
	testingLabels=digitTestingLabels)


	runBayes(1, trainingData=faceTrainingData, validationData=faceValidationData, testingData=faceTestingData,
	trainingLabels=faceTrainingLabels, validationLabels=faceValidationLabels,
	testingLabels=faceTestingLabels)   


	# runDecisionTree(1, faceTrainingData, faceValidationData, faceTestingData, faceTrainingLabels, faceValidationLabels, faceTestingLabels)
	# runDecisionTree(0, digitTrainingData, digitValidationData, digitTestingData, digitTrainingLabels, digitValidationLabels, digitTestingLabels)

def runDecisionTree(i, trainingData, validationData, testingData, 
trainingLabels, validationLabels, testingLabels):

	figure = plt.figure(constrained_layout=True)
	plots = figure.add_gridspec(4, 4)

	hyperparameters = [
		{
			"number" : 0,
			"name": "criterion",
			"dims": plots[0:2, 0],
			"x_label": "criterion",
			"x_content" :["gini", "entropy"],
			"plot_title": "Accuracy of different impurity criteria",
			"values": [
				["gini",None, 2, 1, None],
				["entropy",None, 2, 1, None]
			],
			"stats" : []
		},
		{
			"number" : 1,
			"name": "max depth",			
			"dims":plots[0, 1:4],
			"x_label": "max depth",
			"plot_title":"Accuracy of different max depths",
			"x_content" :list(range(1,20)),
			"values": [["gini", i, 2,1,None] for i in range(1,20)],
			"stats" : [],
		},
		{
			"number" : 2,
			"name": "min samples split",			
			"dims":plots[2, 0:3],
			"x_label": "min samples split",
			"plot_title":"Accuracy of different sample splits",
			"x_content" :list(range(2,20)),
			"values": [["gini", None, i,1,None] for i in range(2,20)],
			"stats" : [],
		},
		{
			"number" : 3,
			"name": "min samples in leaf",			
			"dims":plots[3, 0:3],
			"x_label": "min samples leaf",
			"plot_title":"Accuracy of different samples in leafs",
			"x_content" :list(range(1,20)),
			"values": [["gini", None, 2,i,None] for i in range(1,20)],
			"stats" : [],
		},		
		{
			"number" : 4,
			"name": "max leaf nodes",			
			"dims":plots[1, 1:4],
			"x_label": "max leaf nodes",
			"plot_title":"Accuracy of different leaf nodes",
			"x_content" :list(range(2,20)),
			"values": [["gini", None, 2,1,i] for i in range(2,20)],
			"stats" : [],
		},		

	]

	bestValues = ["",0,0,0,0]

	for hyperparameter in hyperparameters:
		print("Tuning the ",hyperparameter["name"], " hyperparameter")
		maxAccuracy = 0
		for value in hyperparameter["values"]:
			classifier = DecisionTreeClassifier(value[0], value[1], value[2], value[3], value[4])
			print("Training...")
			classifier.train(trainingData=trainingData, trainingLabels=trainingLabels)

			print("Validating...")
			guesses = classifier.classify(testData=validationData)

			accuracy = calcAccuracy(guesses, validationLabels)
			hyperparameter["stats"].append (round(accuracy, 1))
			if (accuracy > maxAccuracy):
				maxAccuracy = accuracy
				bestValues[hyperparameter["number"]] = value[hyperparameter["number"]]

		f = figure.add_subplot(hyperparameter["dims"])

		if hyperparameter["x_label"] == "criterion":
			bars = f.bar(hyperparameter["x_content"], hyperparameter["stats"])
			f.bar_label(bars)
		else:
			x = hyperparameter["x_content"]
			y = hyperparameter["stats"]
			f.plot(x,y)
			f.set_xticks(hyperparameter["x_content"])
			f.set_yticks(list(range(0,151,50)))
			for index in range(len(x)):
				f.text(x[index], y[index], y[index], size=10)


		f.set_title(hyperparameter["plot_title"])
		f.set_xlabel(hyperparameter["x_label"])
		f.set_ylabel("Accuracy")
	plt.show()

	print ("---------------------------------------")
	print ("Best values for each hyperparameter: ")
	print("criterion: ", bestValues[0])
	print("max depth: ", bestValues[1])
	print("minimum number for splitting samples: ", bestValues[2])
	print("minimum samples in a leaf node", bestValues[3])
	print("maximum number of leaf nodes: ", bestValues[4])
	print ("---------------------------------------")

	print("Testing with the tuned hyperparameters...")
	decision_tree.tuned_params[i] = bestValues
	guesses = decision_tree.classify_with_tuned_params(i, trainingData, trainingLabels, testingData)
	calcAccuracy(guesses=guesses, correctLabels=testingLabels)	

				

def runSVM(i, trainingData, validationData, testingData, 
trainingLabels, validationLabels, testingLabels):
	
	figure = plt.figure(constrained_layout=True)
	plots = figure.add_gridspec(3, 1)

	hyperparameters = [
		{
			"number" : 0,
			"name": "gamma",
			"dims": plots[0],
			"x_label": "gamma values",
			"x_content" :[0.0001, 0.001, 0.01, 0.1, 1, 10],
			"plot_title": "Accuracy change with different gamma values",
			"values": [
				[i, 1.0, "rbf"] for i in [0.0001, 0.001, 0.01, 0.1, 1, 10]
			],
			"stats" : []
		},
		{
			"number" : 1,
			"name": "c",
			"dims": plots[1],
			"x_label": "c values",
			"x_content" :[0.01, 0.1, 1, 10, 100],
			"plot_title": "Accuracy change with different c values",
			"values": [
				["scale", i, "rbf"] for i in [0.01, 0.1, 1, 10, 100]
			],
			"stats" : []
		},
		{
			"number" : 2,
			"name": "kernel",
			"dims": plots[2],
			"x_label": "kernels",
			"x_content" :["linear", "poly", "rbf", "sigmoid"],
			"plot_title": "Accuracy change with different kernels",
			"values": [
				["scale", 1.0, i] for i in ["linear", "poly", "rbf", "sigmoid"]
			],
			"stats" : []
		},						
	]
	bestValues = [0,0,""]
	for hyperparameter in hyperparameters:
		print("Tuning the ",hyperparameter["name"], " hyperparameter")
		maxAccuracy = 0
		for value in hyperparameter["values"]:
			classifier = SVMClassifier(value[0], value[1], value[2])
			print("Training...")
			classifier.train(trainingData=trainingData, trainingLabels=trainingLabels)

			print("Validating...")
			guesses = classifier.classify(testData=validationData)

			accuracy = calcAccuracy(guesses, validationLabels)
			hyperparameter["stats"].append (round(accuracy, 1))
			if (accuracy > maxAccuracy):
				maxAccuracy = accuracy
				bestValues[hyperparameter["number"]] = value[hyperparameter["number"]]

		f = figure.add_subplot(hyperparameter["dims"])

		x = hyperparameter["x_content"]
		y = hyperparameter["stats"]
		f.plot(x,y)
		f.set_xticks(hyperparameter["x_content"])
		f.set_yticks(list(range(0,151,50)))
		for index in range(len(x)):
			f.text(x[index], y[index], y[index], size=10)

		f.set_title(hyperparameter["plot_title"])
		f.set_xlabel(hyperparameter["x_label"])
		f.set_ylabel("Accuracy")
	plt.show()

	print ("---------------------------------------")
	print ("Best values for each hyperparameter: ")
	print("gamma ", bestValues[0])
	print("C ", bestValues[1])
	print("kernel ", bestValues[2])

	svm.tuned_params[i] = bestValues

	print("Testing with the tuned hyperparameters...")
	guesses = svm.classify_with_tuned_params(i, trainingData, trainingLabels, testingData)
	calcAccuracy(guesses=guesses, correctLabels=testingLabels)


def runMLP(i, trainingData, validationData, testingData, 
trainingLabels, validationLabels, testingLabels):

	figure = plt.figure(constrained_layout=True)
	plots = figure.add_gridspec(3, 1)

	hyperparameters = [
		{
			"number" : 0,
			"name": "nodes in hidden layers",
			"dims": plots[0],
			"x_label": "hidden leayers",
			"x_content" :["(300, 200, 100)", "(200,100,50)", "(100,50,25)", "(50,25,12)"],
			"plot_title": "Accuracy change with different hidden layers",
			"values": [
				[i,"relu", 0.001] for i in [(300, 200, 100), (200,100,50), (100,50,25), (50,25,12)]
			],
			"stats" : []
		},		
		{
			"number" : 1,
			"name": "activation function",
			"dims": plots[1],
			"x_label": "activation function",
			"x_content" :['identity', 'logistic', 'tanh', 'relu'],
			"plot_title": "Accuracy change with different activation functions",
			"values": [
				[(100,),i, 0.001] for i in ['identity', 'logistic', 'tanh', 'relu']
			],
			"stats" : []
		},
		{
			"number" : 2,
			"name": "learning rate init",
			"dims": plots[2],
			"x_label": "values for learning rate init",
			"x_content" :[0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
			"plot_title": "Accuracy change with different learning rate init",
			"values": [
				[(100,),"relu", i] for i in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
			],
			"stats" : []
		},				
	]
	bestValues = [0,0,0]
	for hyperparameter in hyperparameters:
		print("Tuning the ",hyperparameter["name"], " hyperparameter")
		maxAccuracy = 0
		for value in hyperparameter["values"]:
			classifier = MLP(value[0], value[1], value[2])
			print("Training...")
			classifier.train(trainingData=trainingData, trainingLabels=trainingLabels)

			print("Validating...")
			guesses = classifier.classify(testData=validationData)

			accuracy = calcAccuracy(guesses, validationLabels)
			hyperparameter["stats"].append (round(accuracy, 1))
			if (accuracy > maxAccuracy):
				maxAccuracy = accuracy
				bestValues[hyperparameter["number"]] = value[hyperparameter["number"]]

		f = figure.add_subplot(hyperparameter["dims"])

		x = hyperparameter["x_content"]
		y = hyperparameter["stats"]
		f.plot(x,y)
		if hyperparameter["number"] != 0:
			f.set_xticks(hyperparameter["x_content"])
		f.set_yticks(list(range(0,151,50)))
		for index in range(len(x)):
			f.text(x[index], y[index], y[index], size=10)

		f.set_title(hyperparameter["plot_title"])
		f.set_xlabel(hyperparameter["x_label"])
		f.set_ylabel("Accuracy")
	plt.show()

	print ("---------------------------------------")
	print ("Best values for each hyperparameter: ")
	
	print("hidden layers: ", bestValues[0])
	print("activation function: ", bestValues[1])
	print("learning rate init ", bestValues[2])

	perceptron.tuned_params[i] = bestValues

	print("Testing with the tuned hyperparameters...")
	guesses = perceptron.classify_with_tuned_params(i, trainingData, trainingLabels, testingData)
	calcAccuracy(guesses=guesses, correctLabels=testingLabels)


def runBayes(i, trainingData, validationData, testingData, 
trainingLabels, validationLabels, testingLabels):

	hyperparameters = [
		{
			"number" : 0,
			"name": "Variance smoothing",
			"x_label": "aVariance smoothing values",
			"x_content" :[0.0001, 0.001, 0.01, 0.1],
			"plot_title": "Accuracy change with different var smoothing values",
			"values": [
				[i] for i in [0.0001, 0.001, 0.01, 0.1]
			],
			"stats" : []
		},					
	]
	bestValues = [0]
	for hyperparameter in hyperparameters:
		print("Tuning the ",hyperparameter["name"], " hyperparameter")
		maxAccuracy = 0
		for value in hyperparameter["values"]:
			classifier = BayesClassifier(value[0])
			print("Training...")
			classifier.train(trainingData=trainingData, trainingLabels=trainingLabels)

			print("Validating...")
			guesses = classifier.classify(testData=validationData)

			accuracy = calcAccuracy(guesses, validationLabels)
			hyperparameter["stats"].append (round(accuracy, 1))
			if (accuracy > maxAccuracy):
				maxAccuracy = accuracy
				bestValues[hyperparameter["number"]] = value[hyperparameter["number"]]


		x = hyperparameter["x_content"]
		y = hyperparameter["stats"]
		plt.plot(x,y)
		plt.yticks(list(range(0,151,50)))
		for index in range(len(x)):
			plt.text(x[index], y[index], y[index], size=10)

		plt.title(hyperparameter["plot_title"])
		plt.xlabel(hyperparameter["x_label"])
		plt.ylabel("Accuracy")
	plt.show()

	print ("---------------------------------------")
	print ("Best values for each hyperparameter: ")
	
	print("Variance Smoothing ", bestValues[0])

	bayes.tuned_params[i] = bestValues

	print("Testing with the tuned hyperparameters...")
	guesses = bayes.classify_with_tuned_params(i, trainingData, trainingLabels, testingData)
	calcAccuracy(guesses=guesses, correctLabels=testingLabels)

def runKNN(i, trainingData, validationData, testingData, 
trainingLabels, validationLabels, testingLabels):

	figure = plt.figure(constrained_layout=True)
	plots = figure.add_gridspec(2, 4)

	hyperparameters = [
		{
			"number" : 0,
			"name": "K",
			"dims":plots[0, 1:4],
			"x_label": "K values",
			"x_content" :[i for i in range (2, 21)],
			"plot_title": "Accuracy change with different k values",
			"values": [
				[i, 1] for i in range (2, 21)
			],
			"stats" : []
		},
		{
			"number" : 1,
			"name": "Distance Metric",
			"dims": plots[0:2, 0],
			"x_label": "Distance Metrics",
			"x_content" :["Euclidean", "Manhattan"],
			"plot_title": "Accuracy change with different distance Metrics",
			"values": [
				[1, i] for i in range(1,3)
			],
			"stats" : []			
		}		
	]
	bestValues = [0,""]
	for hyperparameter in hyperparameters:
		print("Tuning the ",hyperparameter["name"], " hyperparameter")
		maxAccuracy = 0
		for value in hyperparameter["values"]:
			classifier = KnnClassifier(value[0], value[1])
			print("Training...")
			classifier.train(trainingData=trainingData, trainingLabels=trainingLabels)

			print("Validating...")
			guesses = classifier.classify(testData=validationData)

			accuracy = calcAccuracy(guesses, validationLabels)
			hyperparameter["stats"].append (round(accuracy, 1))
			if (accuracy > maxAccuracy):
				maxAccuracy = accuracy
				bestValues[hyperparameter["number"]] = value[hyperparameter["number"]]

		f = figure.add_subplot(hyperparameter["dims"])
		if (hyperparameter["number"] == 1):
			bars = f.bar(hyperparameter["x_content"], hyperparameter["stats"])
			f.bar_label(bars)
		else:
			x = hyperparameter["x_content"]
			y = hyperparameter["stats"]
			f.plot(x,y)
			f.set_xticks(hyperparameter["x_content"])
			f.set_yticks(list(range(0,151,50)))
			for index in range(len(x)):
				f.text(x[index], y[index], y[index], size=10)

		f.set_title(hyperparameter["plot_title"])
		f.set_xlabel(hyperparameter["x_label"])
		f.set_ylabel("Accuracy")
	plt.show()

	print ("---------------------------------------")
	print ("Best values for each hyperparameter: ")
	
	print("K ", bestValues[0])
	print("Distance Metric ", bestValues[1])

	knn.tuned_params[i] = bestValues

	print("Testing with the tuned hyperparameters...")
	guesses = knn.classify_with_tuned_params(i, trainingData, trainingLabels, testingData)
	calcAccuracy(guesses=guesses, correctLabels=testingLabels)

if __name__ == '__main__':
	faceTrainingData, faceTrainingLabels,\
	faceValidationData, faceValidationLabels, \
	faceTestingData, faceTestingLabels,\
	digitTrainingData, digitTrainingLabels, \
	digitValidationData, digitValidationLabels, \
	digitTestingData, digitTestingLabels = prepareData()

	# runClassifier(faceTrainingData, faceTrainingLabels,\
	# faceValidationData, faceValidationLabels, \
	# faceTestingData, faceTestingLabels,\
	# digitTrainingData, digitTrainingLabels, \
	# digitValidationData, digitValidationLabels, \
	# digitTestingData, digitTestingLabels)

	run_classifier_with_tuned_parameters(faceTrainingData, faceTrainingLabels,\
	faceTestingData, faceTestingLabels,\
	digitTrainingData, digitTrainingLabels, \
	digitTestingData, digitTestingLabels)	