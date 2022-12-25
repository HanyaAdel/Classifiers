import samples
import util
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



def basicFeatureExtractorDigit(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    """
    a = datum.getPixels()
    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0
    return features


def basicFeatureExtractorFace(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  a = datum.getPixels()

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

    print("digits test" , len(rawDigitTestingData))
    print(len(digitTestingLabels))
    print(len(rawDigitTrainingData))
    print(len(digitTrainingLabels))
    print(len(rawDigitValidationData))
    print(len(digitValidationLabels))

    print ("-----------------------------")
    print(len(rawFaceTestingData))
    print(len(rawFaceTestingLabels))
    print(len(rawFaceTrainingData))
    print(len(faceTrainingLabels))
    print(len(rawFaceValidationData))
    print(len(rawFaceValidationLabels))

    digitFeatureFunction = basicFeatureExtractorDigit
    digitTrainingData = map(digitFeatureFunction, rawDigitTrainingData)
    digitTestingData = map(digitFeatureFunction, rawDigitTestingData)
    digitValidationData = map(digitFeatureFunction, rawDigitValidationData)

    faceFeatureFunction = basicFeatureExtractorFace
    faceTrainingData = map(faceFeatureFunction, rawFaceTrainingData)
    faceTestingData = map(faceFeatureFunction, rawDigitTestingData)
    faceValidationData = map(faceFeatureFunction, rawFaceValidationData)


if __name__ == '__main__':
    runClassifier()
