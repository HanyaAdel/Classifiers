
import matplotlib.pyplot as plt
from sklearn import tree



class DecisionTreeClassifier():

    def __init__(self, criterion, maxDepth):
        self.type = "decisiontree"
        self.classifier = tree.DecisionTreeClassifier(criterion = criterion,
                max_depth= maxDepth)


    def train(self, trainingData, trainingLabels):
    
        # Performing training
        self.classifier.fit(trainingData, trainingLabels)

        plt.figure(figsize=(24,14))
        tree.plot_tree(self.classifier, filled=True, fontsize =12)
        plt.show()
    
    def test(self, testData):
        guesses = []
        for datum in testData:
            guesses.append(self.classifier.predict([datum]))

        return guesses