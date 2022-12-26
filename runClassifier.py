from knn import KnnClassifier
from bayes import NaiveBayesClassifier
import samples
import util
from matplotlib import pyplot as plt
DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28
FACE_DATUM_WIDTH = 60
FACE_DATUM_HEIGHT = 70


TRAINING_DATA_SIZE_DIGITS = 5000
TESTING_DATA_SIZE_DIGITS = 1000
VALIDATION_DATA_SIZE_DIGITS = 1000

TRAINING_DATA_SIZE_FACES = 451 
TESTING_DATA_SIZE_FACES = 150
VALIDATION_DATA_SIZE_FACES = 301



def digitFeatureExtractor(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    """

    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0
    return features


def faceFeatureExtractor(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  features = util.Counter()
  for x in range(FACE_DATUM_WIDTH):
    for y in range(FACE_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def runClassifier():

    rawDigitTrainingData = samples.loadDataFile("digitdata/trainingimages", TRAINING_DATA_SIZE_DIGITS, DIGIT_DATUM_WIDTH,
                                                DIGIT_DATUM_HEIGHT)
    digitTrainingLabels = samples.loadLabelsFile("digitdata/traininglabels", TRAINING_DATA_SIZE_DIGITS)
    rawDigitValidationData = samples.loadDataFile("digitdata/validationimages", VALIDATION_DATA_SIZE_DIGITS, DIGIT_DATUM_WIDTH,
                                                 DIGIT_DATUM_HEIGHT)
    digitValidationLabels = samples.loadLabelsFile("digitdata/validationlabels", VALIDATION_DATA_SIZE_DIGITS)
    rawDigitTestingData = samples.loadDataFile("digitdata/testimages", TESTING_DATA_SIZE_DIGITS, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
    digitTestingLabels = samples.loadLabelsFile("digitdata/testlabels", TESTING_DATA_SIZE_DIGITS)



    rawFaceTrainingData = samples.loadDataFile("facedata/facedatatrain", TRAINING_DATA_SIZE_FACES, FACE_DATUM_WIDTH,
                                               FACE_DATUM_HEIGHT)
    faceTrainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", TRAINING_DATA_SIZE_FACES)
    rawFaceValidationData = samples.loadDataFile("facedata/facedatavalidation", VALIDATION_DATA_SIZE_FACES, FACE_DATUM_WIDTH,
                                                 FACE_DATUM_HEIGHT)
    rawFaceValidationLabels = samples.loadLabelsFile("facedata/facedatavalidationlabels", VALIDATION_DATA_SIZE_FACES)
    rawFaceTestingData = samples.loadDataFile("facedata/facedatatest", TESTING_DATA_SIZE_FACES, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
    rawFaceTestingLabels = samples.loadLabelsFile("facedata/facedatatestlabels", TESTING_DATA_SIZE_FACES)    

    # print("digits test" , len(rawDigitTestingData))
    # print(len(digitTestingLabels))
    # print(len(rawDigitTrainingData))
    # print(len(digitTrainingLabels))
    # print(len(rawDigitValidationData))
    # print(len(digitValidationLabels))

    # print ("-----------------------------")
    # print(len(rawFaceTestingData))
    # print(len(rawFaceTestingLabels))
    # print(len(rawFaceTrainingData))
    # print(len(faceTrainingLabels))
    # print(len(rawFaceValidationData))
    # print(len(rawFaceValidationLabels))


    digitTrainingData = list(map(digitFeatureExtractor, rawDigitTrainingData))
    digitValidationData = list(map(digitFeatureExtractor, rawDigitValidationData))
    digitTestingData = list(map(digitFeatureExtractor, rawDigitTestingData))

    
    faceTrainingData = list(map(faceFeatureExtractor, rawFaceTrainingData))
    faceTestingData = list(map(faceFeatureExtractor, rawFaceTestingData))
    faceValidationData = list(map(faceFeatureExtractor, rawFaceValidationData))              

    digitLegalLabels = range(10)
    faceLegalLabels = range(2)


    runKNN(trainingData=digitTrainingData, validationData=digitValidationData, testingData=digitTestingData,
    legalLabels=digitLegalLabels, trainingLabels=digitTrainingLabels, validationLabels=digitValidationLabels,
    testingLabels=digitTestingLabels)

    # classifier = BayesClassifier(faceLegalLabels)
    # classifier.train(faceTrainingData, faceTrainingLabels, faceValidationData, rawFaceValidationLabels)
    # guesses = classifier.classify(faceValidationData)
    # correct = [guesses[i] == rawFaceValidationLabels[i] for i in range(len(rawFaceValidationLabels))].count(True)
    # print (str(correct), ("correct out of " + str(len(rawFaceValidationLabels)) + " (%.1f%%).") % (100.0 * correct / len(rawFaceValidationLabels)))

    classifier = NaiveBayesClassifier(digitLegalLabels)
    classifier.train(digitTrainingData, digitTrainingLabels, digitValidationData, digitValidationLabels)
    guesses = classifier.classify(digitValidationData)
    correct = [guesses[i] == digitValidationLabels[i] for i in range(len(digitValidationLabels))].count(True)
    print (str(correct), ("correct out of " + str(len(digitValidationLabels)) + " (%.1f%%).") % (100.0 * correct / len(digitValidationLabels)))



def runKNN(trainingData, validationData, testingData, legalLabels, 
trainingLabels, validationLabels, testingLabels):

    plot1 = plt.subplot2grid((5, 5), (0,0))
    plot2 = plt.subplot2grid((5, 5), (0,2))
    Kstats  = []
    for k in range(2, 7):
        classifier = KnnClassifier(legalLabels,k, 0)
        classifier.train(trainingData, trainingLabels, validationData, validationLabels)
        print ("Validating...")
        guesses = classifier.classify(validationData)
        correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
        print("using a k of ", k, " and distance mentric ", 0)
        print (str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels)))
        print ("Testing...")
        guesses = classifier.classify(testingData)
        correct = [guesses[i] == testingLabels[i] for i in range(len(testingLabels))].count(True)
        print (str(correct), ("correct out of " + str(len(testingLabels)) + " (%.1f%%).") % (100.0 * correct / len(testingLabels)))
        accuracy = 100.0 * correct / len(testingLabels)
        Kstats.append((k,round(accuracy, 2)))
    print(Kstats)
    xs = [x[0] for x in Kstats]
    ys = [x[1] for x in Kstats]
    plot1.plot(xs, ys)
    plot1.set_title("Accuracy change with the change of the number of neighbors")
    plot1.set_xlabel("number of neighbors (k)")
    plot1.set_ylabel("Accuracy")

    distanceMetricStats = {"Euclidean" : 0, "Manhattan" : 0}
    classifier = KnnClassifier(legalLabels,4, 0)
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    guesses = classifier.classify(validationData)
    #correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print ("Testing...")
    guesses = classifier.classify(testingData)
    correct = [guesses[i] == testingLabels[i] for i in range(len(testingLabels))].count(True)
    print (str(correct), ("correct out of " + str(len(testingLabels)) + " (%.1f%%).") % (100.0 * correct / len(testingLabels)))
    accuracy = 100.0 * correct / len(testingLabels)
    distanceMetricStats["Euclidean"] = round(accuracy, 3)

    classifier = KnnClassifier(legalLabels,4, 1)
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    guesses = classifier.classify(validationData)
    #correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print ("Testing...")
    guesses = classifier.classify(testingData)
    correct = [guesses[i] == testingLabels[i] for i in range(len(testingLabels))].count(True)
    print (str(correct), ("correct out of " + str(len(testingLabels)) + " (%.1f%%).") % (100.0 * correct / len(testingLabels)))
    accuracy = 100.0 * correct / len(testingLabels)
    distanceMetricStats["Manhattan"] = round(accuracy, 3)    
    plot2.bar(list(distanceMetricStats.keys()), list(distanceMetricStats.values()))

    plot2.set_title("Accuracy of different distance metrics")
    plot2.set_xlabel("Distance Metrics")
    plot2.set_ylabel("Accuracy")

    plt.title("KNN")
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    runClassifier()