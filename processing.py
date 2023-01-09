import samples
import numpy as np

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

def prepareData():
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

	return faceTrainingData, faceTrainingLabels, \
		faceValidationData, faceValidationLabels, \
		faceTestingData, faceTestingLabels,\
		digitTrainingData, digitTrainingLabels,\
		digitValidationData, digitValidationLabels, \
		digitTestingData, digitTestingLabels


def calcAccuracy(guesses, correctLabels):
	correct = [guesses[i] == correctLabels[i] for i in range(len(correctLabels))].count(True)
	print (str(correct), ("correct out of " + str(len(correctLabels)) + " (%.1f%%).") % (100.0 * correct / len(correctLabels)))
	return (100.0 * correct / len(correctLabels))