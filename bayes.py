import util
import math


class BayesClassifier:
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = True # Look at this flag to decide whether to choose k automatically ** use this in your train method **
        self.probs = {}
        self.extra = False

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.
        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.
        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """

        count = util.Counter()
        priorD = util.Counter()

        for label in trainingLabels:
            count[label] += 1
            priorD[label] += 1


        count = {}

        for feat in self.features:

            count[feat] = {}

            for label in self.legalLabels:
                count[feat][label] = {
                    0: 0,
                    1: 0
                }

        for i in range(len(trainingData)):

            datum = trainingData[i]
            label = trainingLabels[i]

            for (feat, val) in datum.items():
                count[feat][label][val] += 1

        #print "priorD normalize" + str(priorD.normalize())

        priorD.normalize()
        self.priorD = priorD


        # Using Laplace smoothing to tune the data and find the best k value that gives the highest accuracy
        bestK = -1
        bestAcc = -1
        # laplace smoothing equation 
        for k in kgrid:
            tempProb = {}
            for (feat, labels) in count.items():
                tempProb[feat] = {}
                for (label, vals) in labels.items():
                    tempProb[feat][label] = {}
                    total = sum(count[feat][label].values())
                    total += 2*k
                    for (val, c) in vals.items():
                        #Normalizing the probability
                        tempProb[feat][label][val] = (count[feat][label][val] + k) / total

            self.probs = tempProb

            predictions = self.classify(validationData)

            # Count number of correct predictions
            acc = 0
            for i in range(len(predictions)):
                if predictions[i] == validationLabels[i]:
                    acc += 1

            # Checking if any of the k values produced the best accuracy and if it did we store it
            if acc > bestAcc:
                bestK = k
                bestAcc = acc
        
        print(bestK)
        print(bestAcc)
        

        #Calculating the probabilities using the best k to get the most accurate results
        tProb = {}
        for (feat, labels) in count.items():
            tProb[feat] = {}

            for (label, vals) in labels.items():
                tProb[feat][label] = {}
                total = sum(count[feat][label].values())
                total += 2*bestK
                for (val, c) in vals.items():
                    tProb[feat][label][val] = (count[feat][label][val] + bestK) / total

        self.probs = tProb

    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.
        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        logJoint = util.Counter()
        #print self.priorD

        for label in self.legalLabels:
            logJoint[label] = math.log(self.priorD[label])
            #print str(logJoint[label])

            for (feat, val) in datum.items():
                #print "inside of datum for loop"
                p = self.probs[feat][label][val];
                logJoint[label] += math.log(p)

        return logJoint

       