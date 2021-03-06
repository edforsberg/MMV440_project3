import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from DataGenerator import generateData
from Preprocessing import deleteFeaturesRandomly


def constantFilter(X_train, constantFilterThreshold):
    constantFilter = VarianceThreshold(threshold=constantFilterThreshold)
    X_train = constantFilter.fit_transform(X_train)
    return X_train

def correlationFilter(X_train, correlationFilterThreshold):
    dataFrame = pd.DataFrame(X_train)
    correlationMatrix = dataFrame.corr().abs()
    upper = correlationMatrix.where(np.triu(np.ones(correlationMatrix.shape), k=1).astype(np.bool))
    featuresToDrop = [column for column in upper.columns if any(upper[column] > correlationFilterThreshold)]
    dataFrame.drop(dataFrame.columns[featuresToDrop], axis=1)
    X_train = np.array(dataFrame)
    return X_train

def fisherScoreFilter(numberOfFeaturesToRemove, X_train, y_train, NUMBER_OF_CLASSES):
    if numberOfFeaturesToRemove == 0:
        return X_train, [], [] 
    
    numberOfFeatures = np.size(X_train, 1)
    classSizes = np.zeros(NUMBER_OF_CLASSES)
    for i in range(NUMBER_OF_CLASSES):
        classSizes[i] = np.count_nonzero(y_train == i)
    classSizes = classSizes.astype(np.int64)

    sizeOfBiggestClass = np.amax(classSizes)
    classIndexes = np.zeros((sizeOfBiggestClass, NUMBER_OF_CLASSES)).astype(np.int64)
    for i in range(NUMBER_OF_CLASSES):
        indexes = np.where(y_train == i)
        for j in range(classSizes[i]):
            classIndexes[j, i] = indexes[0][j]

    featureMeans = np.divide(np.sum(X_train, 1), np.sum(classSizes)) 
    classMeans = np.zeros((numberOfFeatures, NUMBER_OF_CLASSES))
    classVariances = np.zeros((numberOfFeatures, NUMBER_OF_CLASSES))
    for i in range(NUMBER_OF_CLASSES):
        for j in range(numberOfFeatures):
            classMeans[j, i] = np.divide(np.sum(X_train[classIndexes[:, i], j]), classSizes[i])
            diff = classMeans[j, i]-X_train[classIndexes[:, i], j]
            diffSquared = np.dot(diff, diff)
            classVariances[j, i] = np.divide(np.sum(diffSquared), classSizes[i])

    fisherScores = np.zeros(numberOfFeatures)
    for i in range(numberOfFeatures):
        numerator = 0
        denominator = 0
        for j in range(NUMBER_OF_CLASSES):
            numerator += classSizes[j]*(classMeans[i, j]-featureMeans[i])**2
            denominator += classSizes[j]*classVariances[i, j]
        fisherScores[i] = np.divide(numerator, denominator)

    indexes = np.argpartition(fisherScores, numberOfFeaturesToRemove)
    removedFeatures = indexes[0:numberOfFeaturesToRemove]
    X_train = np.delete(X_train, np.s_[removedFeatures], axis=1) 
    return X_train, fisherScores, removedFeatures
