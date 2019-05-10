# -*- coding: utf-8 -*-
"""
Created on Tue May  7 18:07:06 2019

@author: kaany
"""
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

from DataGenerator import generateData
from Preprocessing import transfromFeaturesToNoiseRandomly

from settings import (NUMBER_OF_CLASSES, NUMBER_OF_FEATURES,
                        NUMBER_OF_RECORDS_PER_CLASS,
                        FEATURE_MEAN_RANGE, RANDOM_NUMBER_SEED,
                        NUMBER_OF_FEATURES_TO_PRUNE, TEST_SIZE_PERCENTAGE,
                        NOISE_MEAN, NOISE_STD)

OPACITY = 0.7

np.random.seed(RANDOM_NUMBER_SEED)

data, labels = generateData(NUMBER_OF_CLASSES, NUMBER_OF_FEATURES,
                            NUMBER_OF_RECORDS_PER_CLASS, FEATURE_MEAN_RANGE,
                            RANDOM_NUMBER_SEED)
prunedtrainData = transfromFeaturesToNoiseRandomly(data, labels,
                                                   NUMBER_OF_FEATURES_TO_PRUNE,
                                                   NOISE_MEAN, NOISE_STD,
                                                   randomNumberSeed=RANDOM_NUMBER_SEED)

X_train, X_test, y_train, y_test = train_test_split(prunedtrainData, labels,
                                                    test_size=TEST_SIZE_PERCENTAGE)

distincttrainLabels = np.unique(labels)

# PLOT

plt.figure()
plt.title("Data Set")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
for i, label in enumerate(distincttrainLabels):
    plt.scatter(X_train[y_train==label,0], X_train[y_train==label,1],
                c=np.random.rand(3,), alpha=OPACITY,
                label="Class {}".format(i))

plt.legend()

pca = PCA()
pcaTrainData = pca.fit_transform(X_train)

plt.figure()
plt.title("Feature Selection With PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
for i, label in enumerate(distincttrainLabels):
    plt.scatter(pcaTrainData[y_train==label,0], pcaTrainData[y_train==label,1],
                c=np.random.rand(3,), alpha=OPACITY,
                label="Class {}".format(i))

plt.legend()
