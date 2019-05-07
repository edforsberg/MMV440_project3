import numpy as np
from Initialization import initClasses

def generateData(numberOfClasses, numberOfFeatures, numberOfRecordsPerClass,
                 featureMeanRange, randomNumberSeed):
    TOTAL_NUMBER_OF_RECORDS = numberOfClasses * numberOfRecordsPerClass
            
    featureDistributionData = initClasses(numberOfClasses, numberOfFeatures,
                                            featureMeanRange,
                                            randomNumberSeed=randomNumberSeed)
    
    # Initialize data nxp
    data = np.empty((TOTAL_NUMBER_OF_RECORDS, numberOfFeatures))
    for i in range(numberOfClasses):
        rowRangeForCurrentClass = range(i*200,(i+1)*200)
        generatedFeatures = np.random.multivariate_normal(featureDistributionData[i].featureMeans,
                                                        featureDistributionData[i].featureCovariances,
                                                        size=(numberOfRecordsPerClass))
        data[rowRangeForCurrentClass,:] = generatedFeatures
    
    labels = list(range(numberOfClasses))
    labelsArray = np.repeat(labels, numberOfRecordsPerClass)
    
    return (data, labelsArray)

if __name__ == "__main__":
    
    NUMBER_OF_CLASSES = 4
    NUMBER_OF_FEATURES = NUMBER_OF_CLASSES*2
    NUMBER_OF_FEATURES_PER_CLASS = 200
    TOTAL_NUMBER_OF_RECORDS = NUMBER_OF_CLASSES * NUMBER_OF_FEATURES_PER_CLASS
        
    FEATURE_MEAN_RANGE = [0, 50]
        
    RANDOM_NUMBER_SEED = 0
    
    data, labels = generateData(NUMBER_OF_CLASSES, NUMBER_OF_FEATURES,
                                NUMBER_OF_FEATURES_PER_CLASS, FEATURE_MEAN_RANGE,
                                RANDOM_NUMBER_SEED)