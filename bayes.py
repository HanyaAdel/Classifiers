from sklearn.naive_bayes import GaussianNB

tuned_params = [[0.1], [0.0001]]

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

def classify_with_tuned_params(i, trainingData, trainingLabels, testingData):
    classifier = BayesClassifier(tuned_params[i][0])
    classifier.train(trainingData, trainingLabels)
    return classifier.classify(testingData)