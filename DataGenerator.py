import numpy as np
from Initialization import initClasses

def generateData():
    NUMBER_OF_CLASSES = 4
    NUMBER_OF_FEATURES = NUMBER_OF_CLASSES*2
    NUMBER_OF_RECORDS_PER_CLASS = 200
    TOTAL_NUMBER_OF_RECORDS = NUMBER_OF_CLASSES * NUMBER_OF_RECORDS_PER_CLASS
    
    FEATURE_MEAN_RANGE = [0, 50]
    
    RANDOM_NUMBER_SEED = 0
    
    featureDistributionData = initClasses(NUMBER_OF_CLASSES, NUMBER_OF_FEATURES,
                                            FEATURE_MEAN_RANGE,
                                            randomNumberSeed=RANDOM_NUMBER_SEED)
    
    data = np.empty((TOTAL_NUMBER_OF_RECORDS, NUMBER_OF_FEATURES))
    for i in range(NUMBER_OF_CLASSES):
        rowRangeForCurrentClass = range(i*200,(i+1)*200)
        generatedFeatures = np.random.multivariate_normal(featureDistributionData[i].featureMeans,
                                                        featureDistributionData[i].featureCovariances,
                                                        size=(NUMBER_OF_RECORDS_PER_CLASS))
        data[rowRangeForCurrentClass,:] = generatedFeatures
    
    labels = list(range(NUMBER_OF_CLASSES))
    labelsArray = np.repeat(labels, NUMBER_OF_RECORDS_PER_CLASS)
    
    return (data, labelsArray)

if __name__ == "__main__":
    data, labels = generateData()