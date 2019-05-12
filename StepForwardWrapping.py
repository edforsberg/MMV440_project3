# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:10:49 2019

@author: ErikF (and Kaan) :)
"""

# =============================================================================
# import numpy as np
# from mlxtend.feature_selection import SequentialFeatureSelector
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.decomposition import PCA
# 
# def StepForwardWrapping(data, labels, nrFeatures, k = 5): 
#     
#     nrClasses = len(set(labels))
#     nrDataPts, nrFeaturesOriginal = data.shape
#    # nrDataPoints = data.shape[0]
#     features = []
#     feature_selector = SequentialFeatureSelector(KNeighborsClassifier(5),
#                k_features=k,
#                forward=True,
#                verbose=0,
#                cv=5,
#                n_jobs=-1)
# 
#     features = feature_selector.fit(data, labels)
#     return features 
# 
# def StepForwardWrappingPCA(data, labels, nrFeatures, k = 5):
#     pca = PCA()
# =============================================================================
    
    
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from DataGenerator import generateData
from Preprocessing import transfromFeaturesToNoiseRandomly
from time import time

from settings import (NUMBER_OF_CLASSES, NUMBER_OF_FEATURES,
                        NUMBER_OF_RECORDS_PER_CLASS,
                        FEATURE_MEAN_RANGE, NUMBER_OF_FEATURES_TO_PRUNE,
                        NOISE_MEAN, NOISE_STD,
                        TEST_SIZE_PERCENTAGE)

RANDOM_NUMBER_SEEDS = range(0,20)
NUMBER_OF_NON_NOISY_FEATURES = NUMBER_OF_FEATURES - NUMBER_OF_FEATURES_TO_PRUNE

NUMBER_OF_FEATURES_TO_SELECT_RANGE = range(1,NUMBER_OF_FEATURES)


def runWrappingAndGetAccuracies(randomNumberSeed, nFeaturesToSelect):
    np.random.seed(randomNumberSeed)

    data, labels = generateData(NUMBER_OF_CLASSES, NUMBER_OF_FEATURES,
                                NUMBER_OF_RECORDS_PER_CLASS, FEATURE_MEAN_RANGE,
                                randomNumberSeed)

    trainData = transfromFeaturesToNoiseRandomly(data, labels,
                                     NUMBER_OF_FEATURES_TO_PRUNE,
                                     NOISE_MEAN, NOISE_STD,
                                     randomNumberSeed=randomNumberSeed)

    X_train, X_test, y_train, y_test = train_test_split(trainData, labels,
                                                        test_size=TEST_SIZE_PERCENTAGE)

    n_neighbors = 5

    feature_selector = SequentialFeatureSelector(KNeighborsClassifier(n_neighbors),
               k_features=nFeaturesToSelect,
               forward=True,
               verbose=0,
               cv=5,
               n_jobs=-1)

    features = feature_selector.fit(X_train, y_train)

    xTrainWithSelectedFeatures = X_train[:, features.k_feature_idx_]
    xTestWithSelectedFeatures = X_test[:, features.k_feature_idx_]

    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(xTrainWithSelectedFeatures, y_train)

    train_pred = knn.predict(xTrainWithSelectedFeatures)
    accuracyTrain = accuracy_score(y_train, train_pred)

    test_pred = knn.predict(xTestWithSelectedFeatures)
    accuracyTest = accuracy_score(y_test, test_pred)

    return (accuracyTrain, accuracyTest)

class AccuracyData:

    def __init__(self, meanTrain, stdTrain, meanTest, stdTest, meanTime):
        self.meanTrain = meanTrain
        self.stdTrain = stdTrain
        self.meanTest = meanTest
        self.stdTest = stdTest
        self.meanTime = meanTime

meanTrainAccuracies = []
meanTestAccuracies = []
stdTrainAccuracies = []
stdTestAccuracies = []

for nFeatures in NUMBER_OF_FEATURES_TO_SELECT_RANGE:
    
    if nFeatures == 3:
        c = 211

    trainAccuracies = []
    testAccuracies = []

    durations = []

    for seed in RANDOM_NUMBER_SEEDS:
        a = time()
        trainAccuracy, testAccuracy = runWrappingAndGetAccuracies(seed, nFeatures)
        b = time()
        trainAccuracies.append(trainAccuracy)
        testAccuracies.append(testAccuracy)
        durations.append(b-a)

    meanTrainAccuracy = np.mean(trainAccuracies)
    stdTrainAccuracy = np.std(trainAccuracies)

    meanTestAccuracy = np.mean(testAccuracies)
    stdTestAccuracy = np.std(testAccuracies)

    meanTime = np.mean(durations)

    meanTrainAccuracies.append(meanTrainAccuracy)
    meanTestAccuracies.append(meanTestAccuracy)
    stdTrainAccuracies.append(stdTrainAccuracy)
    stdTestAccuracies.append(stdTestAccuracy)
    durations.append(meanTime)

meanDuration = np.mean(durations)


meanTrainAccuracies.reverse()
stdTrainAccuracies.reverse()
meanTestAccuracies.reverse()
stdTestAccuracies.reverse()

plt.figure()
#plt.errorbar(NUMBER_OF_FEATURES_TO_SELECT_RANGE, meanTrainAccuracies,
#             yerr=stdTrainAccuracies, label="Training Set",
#             fmt='_', capthick=2, capsize=10)
plt.errorbar(NUMBER_OF_FEATURES_TO_SELECT_RANGE, meanTestAccuracies,
             yerr=stdTestAccuracies, label="Test data",
             capthick=2, capsize=10)
plt.errorbar(NUMBER_OF_FEATURES_TO_SELECT_RANGE, meanTrainAccuracies,
             yerr=stdTrainAccuracies, label="Training data",
             capthick=2, capsize=10)
plt.title("Number Of Features to remove vs Accuracy" +
          "Number Of Non-Noisy Features: {}".format(NUMBER_OF_NON_NOISY_FEATURES))
plt.xlabel("Number Of Features to remove")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

saveData = AccuracyData(meanTrainAccuracies, stdTrainAccuracies,
                        meanTestAccuracies, stdTestAccuracies,
                        meanDuration)
np.save("ForwardWrappingMeanAndStdData", saveData)
        
