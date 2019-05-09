# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:12:18 2019

@author: kaany
"""

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector
from matplotlib import pyplot as plt

from DataGenerator import generateData
from Preprocessing import transfromFeaturesToNoiseRandomly

NUMBER_OF_CLASSES = 6
NUMBER_OF_FEATURES = NUMBER_OF_CLASSES*2
NUMBER_OF_FEATURES_PER_CLASS = 500

FEATURE_MEAN_RANGE = [0, 10]

RANDOM_NUMBER_SEEDS = range(0,20)
NUMBER_OF_FEATURES_TO_PRUNE = int(NUMBER_OF_FEATURES / 2)
NUMBER_OF_NON_NOISY_FEATURES = NUMBER_OF_FEATURES - NUMBER_OF_FEATURES_TO_PRUNE

TEST_SIZE_PERCENTAGE = 0.2

NUMBER_OF_FEATURES_TO_SELECT_RANGE = range(1, NUMBER_OF_FEATURES)

def runWrappingAndGetAccuracies(randomNumberSeed, nFeaturesToSelect):
    np.random.seed(randomNumberSeed)

    NOISE_MEAN = np.random.rand() * FEATURE_MEAN_RANGE[1] - FEATURE_MEAN_RANGE[0]
    NOISE_STD = np.random.rand() * FEATURE_MEAN_RANGE[1] - FEATURE_MEAN_RANGE[0]

    data, labels = generateData(NUMBER_OF_CLASSES, NUMBER_OF_FEATURES,
                                NUMBER_OF_FEATURES_PER_CLASS, FEATURE_MEAN_RANGE,
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
               forward=False,
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

    def __init__(self, meanTrain, stdTrain, meanTest, stdTest):
        self.meanTrain = meanTrain
        self.stdTrain = stdTrain
        self.meanTest = meanTest
        self.stdTest = stdTest

meanTrainAccuracies = []
meanTestAccuracies = []
stdTrainAccuracies = []
stdTestAccuracies = []

for nFeatures in NUMBER_OF_FEATURES_TO_SELECT_RANGE:

    trainAccuracies = []
    testAccuracies = []
    for seed in RANDOM_NUMBER_SEEDS:
        trainAccuracy, testAccuracy = runWrappingAndGetAccuracies(seed, nFeatures)

        trainAccuracies.append(trainAccuracy)
        testAccuracies.append(testAccuracy)

    meanTrainAccuracy = np.mean(trainAccuracies)
    stdTrainAccuracy = np.std(trainAccuracies)

    meanTestAccuracy = np.mean(testAccuracies)
    stdTestAccuracy = np.std(testAccuracies)

    meanTrainAccuracies.append(meanTrainAccuracy)
    meanTestAccuracies.append(meanTestAccuracy)
    stdTrainAccuracies.append(stdTrainAccuracy)
    stdTestAccuracies.append(stdTestAccuracy)

plt.figure()
#plt.errorbar(NUMBER_OF_FEATURES_TO_SELECT_RANGE, meanTrainAccuracies,
#             yerr=stdTrainAccuracies, label="Training Set",
#             fmt='_', capthick=2, capsize=10)
plt.errorbar(NUMBER_OF_FEATURES_TO_SELECT_RANGE, meanTestAccuracies,
             yerr=stdTestAccuracies, label="Test Set",
             capthick=2, capsize=10)
plt.title("Number Of Features to Select vs Accuracy\n" +
          "Number Of Noisy Non-Noisy Features: {}".format(NUMBER_OF_NON_NOISY_FEATURES))
plt.xlabel("Number Of Features to Select")
plt.ylabel("Accuracy")
plt.legend()

saveData = AccuracyData(meanTrainAccuracies, stdTrainAccuracies,
                        meanTestAccuracies, stdTestAccuracies)
np.save("BackwardWrappingMeanAndStdData", saveData)

