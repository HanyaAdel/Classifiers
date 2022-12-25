import sys

def readCommand(argv):
    """Processes the command used to run from the command line."""
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-r', '--run',  help='automatically runs training and test cycle for 5 times',
                      default= False, action='store_true')

    parser.add_option('-c', '--classifier', help='The type of classifier',
                      choices=['perceptron', 'naiveBayes', 'mira'],
                      default='naiveBayes')
    parser.add_option('-d', '--data', help='Dataset to use', choices=['digits', 'faces'], default='digits')

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print("Doing classification")
    print("--------------------")
    print("data:\t\t" + options.data)
    print("classifier:\t\t" + options.classifier)

    if options.data == "digits":
        featureFunction = basicFeatureExtractorDigit
    elif options.data == "faces":
        featureFunction = basicFeatureExtractorFace
    else:
        print("Unknown dataset", options.data)
        print(USAGE_STRING)
        sys.exit(2)

    if options.data == "digits":
        legalLabels = range(10)
    else:
        legalLabels = range(2)


    if options.classifier == "naiveBayes":
        classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
    elif options.classifier == "perceptron":
        classifier = perceptron.PerceptronClassifier(legalLabels)
    elif options.classifier == "knn":
        classifier = knn.KNNClassifier(legalLabels)
    elif options.classifier == "mlp":
        classifier = mlp.MLPClassifier(legalLabels)     
    elif options.classifier == "decisionTree":
        classifier = decisionTree.Decision()                        

    else:
        print("Unknown classifier:", options.classifier)
        print(USAGE_STRING)

        sys.exit(2)

    args['classifier'] = classifier
    args['featureFunction'] = featureFunction
    return args, options


USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """


if __name__ == '__main__':
    # Read input
    args, options = readCommand( sys.argv[1:] ) 
    # Run classifier
    runClassifier(args, options)
