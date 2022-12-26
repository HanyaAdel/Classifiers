import util
import math


class BayesClassifier:
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.alpha = 1    # this is the default laplacian smoothing parameter
        self.automaticTuning = False     #Flag to decide whether to choose alpha automatically 
        self.probs = {}
        self.extra = False

    def setSmoothing(self, alpha):
        self.alpha = alpha

    def train(self, trainingData, trainingLabels, validationData, validationLabels):

        self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));

        if (self.automaticTuning):
            alpha_vals = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            alpha_vals = [self.alpha]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, alpha_vals)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, alpha_vals):
       
        count = util.Counter()
        prior = util.Counter()

        for label in trainingLabels:
            count[label] += 1
            prior[label] += 1

        count = {}

        for features in self.features:

            count[features] = {}

            for label in self.legalLabels:
                count[features][label] = {
                    0: 0,
                    1: 0
                }

        for i in range(len(trainingData)):

            datum = trainingData[i]
            label = trainingLabels[i]

            for (features, val) in datum.items():
                count[features][label][val] += 1


        prior.normalize()
        self.prior = prior

        prior[0] = 0.9
        prior[1] = 0.1
        print(prior)
        '''
        prior[0] = 0.07
        prior[1] = 0.02
        prior[2] = 0.03
        prior[3] = 0.08
        prior[4] = 0.05
        prior[5] = 0.04
        prior[6] = 0.05
        prior[7] = 0.06
        prior[8] = 0.01
        prior[9] = 0.09
        '''

        # Using Laplace smoothing to tune the data and find the best alpha value that gives the highest accuracy
        bestAlpha = -1
        bestAccuracy = -1
        # laplace smoothing equation 
        for alpha in alpha_vals:
            tempProb = {}
            for (features, labels) in count.items():
                tempProb[features] = {}
                for (label, vals) in labels.items():
                    tempProb[features][label] = {}
                    total = sum(count[features][label].values())
                    total += 2*alpha
                    for (val, c) in vals.items():
                        #Normalizing the probability
                        tempProb[features][label][val] = (count[features][label][val] + alpha) / total

            self.probs = tempProb

            predictions = self.classify(validationData)

            # Count number of correct predictions
            accuracy = 0
            for i in range(len(predictions)):
                if predictions[i] == validationLabels[i]:
                    accuracy += 1

            # Checking if any of the alpha values produced the best accuracy and if it did we store it
            if accuracy > bestAccuracy:
                bestAlpha = alpha
                bestAccuracy = accuracy
        
        print(bestAlpha)
        print(bestAccuracy)
        

        #Calculating the probabilities using the best alpha to get the most accurate results
        tProb = {}
        for (features, labels) in count.items():
            tProb[features] = {}

            for (label, vals) in labels.items():
                tProb[features][label] = {}
                total = sum(count[features][label].values())
                total += 2*bestAlpha
                for (val, c) in vals.items():
                    tProb[features][label][val] = (count[features][label][val] + bestAlpha) / total

        self.probs = tProb

    def classify(self, testData):

        guesses = []
        self.posteriors = [] 
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        logJoint = util.Counter()

        for label in self.legalLabels:
            logJoint[label] = math.log(self.prior[label])

            for (features, val) in datum.items():
                p = self.probs[features][label][val];
                logJoint[label] += math.log(p)

        return logJoint

       