from knn import KnnClassifier
from bayes import BayesClassifier
import samples
import util
import numpy as np
from matplotlib import pyplot as plt

DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28
FACE_DATUM_WIDTH = 60
FACE_DATUM_HEIGHT = 60



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
    faceValidationLabels = samples.loadLabelsFile("facedata/facedatavalidationlabels", VALIDATION_DATA_SIZE_FACES)
    rawFaceTestingData = samples.loadDataFile("facedata/facedatatest", TESTING_DATA_SIZE_FACES, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
    faceTestingLabels = samples.loadLabelsFile("facedata/facedatatestlabels", TESTING_DATA_SIZE_FACES)    


    digitTrainingData = []
    for i in range (len(rawDigitTrainingData)):
      digitTrainingData.append(rawDigitTrainingData[i].getPixels())

    digitTestingData = []
    for i in range (len(rawDigitTestingData)):
      digitTestingData.append(rawDigitTestingData[i].getPixels())    

    digitValidationData = []
    for i in range (len(rawDigitValidationData)):
      digitValidationData.append(rawDigitValidationData[i].getPixels())            


    digitTrainingData = np.array(digitTrainingData).reshape(
      TRAINING_DATA_SIZE_DIGITS, DIGIT_DATUM_WIDTH * DIGIT_DATUM_HEIGHT)
    digitTrainingLabels = np.array(digitTrainingLabels)

    digitTestingData = np.array(digitTestingData).reshape(
      TESTING_DATA_SIZE_DIGITS, DIGIT_DATUM_WIDTH * DIGIT_DATUM_HEIGHT)    
    digitValidationData = np.array(digitValidationData).reshape(
      VALIDATION_DATA_SIZE_DIGITS, DIGIT_DATUM_WIDTH * DIGIT_DATUM_HEIGHT)    


    faceTrainingData = []
    for i in range (len(rawFaceTrainingData)):
      faceTrainingData.append(rawFaceTrainingData[i].getPixels())
    
    faceValidationData = []
    for i in range (len(rawFaceValidationData)):
      faceValidationData.append(rawFaceValidationData[i].getPixels())

    faceTestingData = []
    for i in range (len(rawFaceTestingData)):
      faceTestingData.append(rawFaceTestingData[i].getPixels())

    faceTrainingData = np.array(faceTrainingData).reshape(
      TRAINING_DATA_SIZE_FACES, FACE_DATUM_WIDTH * FACE_DATUM_HEIGHT)      
    faceValidationData = np.array(faceValidationData).reshape(
      VALIDATION_DATA_SIZE_FACES, FACE_DATUM_WIDTH * FACE_DATUM_HEIGHT)            
    faceTestingData = np.array(faceTestingData).reshape(
      TESTING_DATA_SIZE_FACES, FACE_DATUM_WIDTH * FACE_DATUM_HEIGHT) 

    faceTrainingLabels = np.array(faceTrainingLabels)
    faceValidationLabels = np.array(faceValidationLabels)
    faceTestingLabels = np.array(faceTestingLabels)    


    # runKNN(trainingData=digitTrainingData, validationData=digitValidationData, testingData=digitTestingData,
    # trainingLabels=digitTrainingLabels, validationLabels=digitValidationLabels,
    # testingLabels=digitTestingLabels)
    
    # runKNN(trainingData=faceTrainingData, validationData=faceValidationData, testingData=faceTestingData,
    # trainingLabels=faceTrainingLabels, validationLabels=faceValidationLabels,
    # testingLabels=faceTestingLabels)    

    # runBayes(trainingData=digitTrainingData, validationData=digitValidationData, testingData=digitTestingData,
    # trainingLabels=digitTrainingLabels, validationLabels=digitValidationLabels,
    # testingLabels=digitTestingLabels)

    runBayes(trainingData=faceTrainingData, validationData=faceValidationData, testingData=faceTestingData,
    trainingLabels=faceTrainingLabels, validationLabels=faceValidationLabels,
    testingLabels=faceTestingLabels)    



def runBayes(trainingData, validationData, testingData, 
trainingLabels, validationLabels, testingLabels):
  alpha_set = [0.0001, 0.001, 0.01, 0.1]

  alpha_stats = []

  for i in range(len(alpha_set)):
    classifier = BayesClassifier(alpha_set[i])
    print("variance smoothing value: ", alpha_set[i])

    print("Training....")
    classifier.train(trainingData, trainingLabels)

    print("Validating....")
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print (str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels)))
    
    print("Testing....")
    guesses = classifier.classify(testingData)
    correct = [guesses[i] == testingLabels[i] for i in range(len(testingLabels))].count(True)
    print (str(correct), ("correct out of " + str(len(testingLabels)) + " (%.1f%%).") % (100.0 * correct / len(testingLabels)))

    accuracy = 100.0 * correct / len(testingLabels)
    alpha_stats.append((alpha_set[i],round(accuracy, 2)))
  xs = [x[0] for x in alpha_stats]
  ys = [x[1] for x in alpha_stats]
  plt.xticks(alpha_set)
  plt.plot(xs, ys)
  plt.title("Accuracy change with the change of variance smoothing")
  plt.xlabel("variance smoothing value")
  plt.ylabel("Accuracy")    
  plt.show()


def runKNN(trainingData, validationData, testingData, 
trainingLabels, validationLabels, testingLabels):
  figure, axis = plt.subplots(1, 2)
  Kstats  = []
  for k in range(2, 41):
    print("K = ", k)

    classifier = KnnClassifier(k, 1)

    print ("Training...")
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)

    print ("Validating...")
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print("using a k of ", k, " and distance metric Euclidean")
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
  axis[0].plot(xs, ys)
  axis[0].set_title("Accuracy change with the change of the number of neighbors")
  axis[0].set_xlabel("number of neighbors (k)")
  axis[0].set_ylabel("Accuracy")

  distanceMetricStats = {1 : 0, 2: 0}

  for i in range (1,3):
    classifier = KnnClassifier(4, i)
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print ("Testing...")
    guesses = classifier.classify(testingData)
    correct = [guesses[i] == testingLabels[i] for i in range(len(testingLabels))].count(True)
    print (str(correct), ("correct out of " + str(len(testingLabels)) + " (%.1f%%).") % (100.0 * correct / len(testingLabels)))
    accuracy = 100.0 * correct / len(testingLabels)
    distanceMetricStats[i] = round(accuracy, 3)


  axis[1].bar(["Euclidean", "Manhattan"], list(distanceMetricStats.values()))

  axis[1].set_title("Accuracy of different distance metrics")
  axis[1].set_xlabel("Distance Metrics")
  axis[1].set_ylabel("Accuracy")

  plt.show()


if __name__ == '__main__':
    runClassifier()