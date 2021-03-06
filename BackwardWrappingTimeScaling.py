# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:12:18 2019

@author: kaany
"""

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
from matplotlib import pyplot as plt

from DataGenerator import generateData
from Preprocessing import transfromFeaturesToNoiseRandomly
from time import time

from settings import (NUMBER_OF_CLASSES, NUMBER_OF_RECORDS_PER_CLASS,
                        FEATURE_MEAN_RANGE,
                        NOISE_MEAN, NOISE_STD,
                        TEST_SIZE_PERCENTAGE)

RANDOM_NUMBER_SEEDS = range(0,20)
NUMBER_OF_FEATURES = range(2,20)

def runWrappingAndGetAccuraciesWithPCA(randomNumberSeed, nFeatures, nFeaturesToSelect):
    np.random.seed(randomNumberSeed)

    data, labels = generateData(NUMBER_OF_CLASSES, nFeatures,
                                NUMBER_OF_RECORDS_PER_CLASS, FEATURE_MEAN_RANGE,
                                randomNumberSeed)

    NUMBER_OF_FEATURES_TO_PRUNE = nFeatures - nFeaturesToSelect
    trainData = transfromFeaturesToNoiseRandomly(data, labels,
                                     NUMBER_OF_FEATURES_TO_PRUNE,
                                     NOISE_MEAN, NOISE_STD,
                                     randomNumberSeed=randomNumberSeed)

    X_train, X_test, y_train, y_test = train_test_split(trainData, labels,
                                                        test_size=TEST_SIZE_PERCENTAGE)

    n_neighbors = 5
    a = time()
    feature_selector = SequentialFeatureSelector(KNeighborsClassifier(n_neighbors),
               k_features=nFeaturesToSelect,
               forward=False,
               verbose=0,
               cv=5,
               n_jobs=-1)
    _ = feature_selector.fit(X_train, y_train)
    b = time()
    return (b-a)

meanDurations = []
stdDurations = []

for nFeatures in NUMBER_OF_FEATURES:

    nFeaturesToSelect = int(nFeatures/2)
    durations = []

    for seed in RANDOM_NUMBER_SEEDS:

        duration = runWrappingAndGetAccuraciesWithPCA(seed, nFeatures, nFeaturesToSelect)
        durations.append(duration)

    meanTime = np.mean(durations)
    stdTime = np.std(durations)

    meanDurations.append(meanTime)
    stdDurations.append(stdTime)

plt.figure(figsize=(8,6))
plt.errorbar(NUMBER_OF_FEATURES, meanDurations,
             yerr=stdDurations, label="Mean Duration",
             capthick=2, capsize=10)
plt.title("Backward Elimination\nNumber Of Features in Data Set vs Duration")
plt.xlabel("Number Of Dimenions in Data Set")
plt.ylabel("Duration (Seconds)")
plt.legend()
plt.show()

