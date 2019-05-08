import numpy as np
from Class import Class

def initClasses(numberOfClasses, numberOfFeatures, featureMeanRange,
                            randomNumberSeed=None):
    featureMeansOfEachClass = initializeFeatureMeans(numberOfClasses,
                                                    numberOfFeatures,
                                                    featureMeanRange,
                                                    randomNumberSeed=randomNumberSeed)

    featureCovsOfEachClass = initializeFeatureCov(numberOfClasses,
                                                    numberOfFeatures,
                                                    randomNumberSeed=randomNumberSeed)

    featureDistributionData = np.empty((numberOfClasses,), dtype=Class)
    for i in range(numberOfClasses):
        featureDistributionData[i] = Class(featureMeansOfEachClass[i],
                                            featureCovsOfEachClass[i])

    return featureDistributionData

def initializeFeatureMeans(numberOfClasses, numberOfFeatures, _range,
                            randomNumberSeed=None):

    if randomNumberSeed != None:
        np.random.seed(randomNumberSeed)

    # Initialize each feature's mean for each class
    featureMeansOfEachClass = np.empty((numberOfClasses, numberOfFeatures))
    for i in range(numberOfClasses):
        initialMeans = np.random.rand(numberOfFeatures)
        featureMeansOfEachClass[i] = initialMeans*_range[1] - _range[0]
    return featureMeansOfEachClass

def initializeFeatureCov(numberOfClasses, numberOfFeatures, randomNumberSeed=None):

    if randomNumberSeed != None:
        np.random.seed(randomNumberSeed)

    # Initialize feature covariance for each class
    featureCovarianceOfEachClass = np.empty((numberOfClasses, numberOfFeatures, numberOfFeatures))
    for i in range(numberOfClasses):
        initialCov = np.random.rand(numberOfFeatures, numberOfFeatures)
        
        # Make positive semi-definite
        initialCov = np.dot(initialCov,initialCov.transpose())
        #Make diag 1
        np.fill_diagonal(initialCov, 1)
        #Make symmetric
        symmetricCov = np.maximum(initialCov, initialCov.transpose())
        
        featureCovarianceOfEachClass[i] = symmetricCov
    return featureCovarianceOfEachClass

def removeFeatureFromTestSet(testData, testLabels, trainData, trainLabels):
    labels = np.unique(trainLabels)
    
    for label in labels:
        pass
        
    
    
    
    