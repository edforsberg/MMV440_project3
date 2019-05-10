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

from settings import (NUMBER_OF_CLASSES, NUMBER_OF_FEATURES,
                        NUMBER_OF_RECORDS_PER_CLASS,
                        FEATURE_MEAN_RANGE, NUMBER_OF_FEATURES_TO_PRUNE,
                        NOISE_MEAN, NOISE_STD,
                        TEST_SIZE_PERCENTAGE)

NUMBER_OF_FEATURES_TO_SELECT = 3
RANDOM_NUMBER_SEEDS = range(0,20)

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
               forward=False,
               verbose=0,
               cv=5,
               n_jobs=-1)

    features = feature_selector.fit(X_train, y_train)
    selectedFeatureSubset = features.k_feature_idx_

    xTrainWithSelectedFeatures = X_train[:, selectedFeatureSubset]
    xTestWithSelectedFeatures = X_test[:, selectedFeatureSubset]

    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(xTrainWithSelectedFeatures, y_train)

    train_pred = knn.predict(xTrainWithSelectedFeatures)
    accuracyTrain = accuracy_score(y_train, train_pred)

    test_pred = knn.predict(xTestWithSelectedFeatures)
    accuracyTest = accuracy_score(y_test, test_pred)

    return (accuracyTrain, accuracyTest, selectedFeatureSubset)

trainAccuracies = []
testAccuracies = []
selectedFeatureSubsets = []

for seed in RANDOM_NUMBER_SEEDS:
    trainAccuracy, testAccuracy, selectedFeatureSubset = runWrappingAndGetAccuracies(seed,
                                                                                     NUMBER_OF_FEATURES_TO_SELECT)
    trainAccuracies.append(trainAccuracy)
    testAccuracies.append(testAccuracy)
    selectedFeatureSubsets.append(selectedFeatureSubset)

for i in range(len(testAccuracies)):
    print("Selected Feature Subset: {}\tTest Accuracy: {:.2f}".format(selectedFeatureSubsets[i],
                                                                  testAccuracies[i]))




