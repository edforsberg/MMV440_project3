# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:10:48 2019

@author: kaany
"""

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

from DataGenerator import generateData
from Preprocessing import transfromFeaturesToNoiseRandomly

NUMBER_OF_CLASSES = 6
NUMBER_OF_FEATURES = NUMBER_OF_CLASSES*2
NUMBER_OF_FEATURES_PER_CLASS = 500
TOTAL_NUMBER_OF_RECORDS = NUMBER_OF_CLASSES * NUMBER_OF_FEATURES_PER_CLASS

FEATURE_MEAN_RANGE = [0, 10]

RANDOM_NUMBER_SEED = 0
NUMBER_OF_FEATURES_TO_PRUNE = int(NUMBER_OF_FEATURES / 2)

OPACITY = 0.7

NOISE_MEAN = 10
NOISE_STD = 5

TEST_SIZE_PERCENTAGE = 0.2

np.random.seed(RANDOM_NUMBER_SEED)

data, labels = generateData(NUMBER_OF_CLASSES, NUMBER_OF_FEATURES,
                            NUMBER_OF_FEATURES_PER_CLASS, FEATURE_MEAN_RANGE,
                            RANDOM_NUMBER_SEED)
trainData = transfromFeaturesToNoiseRandomly(data, labels,
                                           NUMBER_OF_FEATURES_TO_PRUNE,
                                           NOISE_MEAN, NOISE_STD,
                                           randomNumberSeed=RANDOM_NUMBER_SEED)

X_train, X_test, y_train, y_test = train_test_split(trainData, labels,
                                                    test_size=TEST_SIZE_PERCENTAGE)

distincttrainLabels = np.unique(labels)

# PLOT

#plt.figure()
#plt.title("Data Set")
#plt.xlabel("Feature 1")
#plt.ylabel("Feature 2")
#for i, label in enumerate(distincttrainLabels):
#    _ = np.random.rand() # This avoids the strange thing the random number
#    # generator does while generating new colors for scatter plot
#    plt.scatter(X_train[y_train==label,0], X_train[y_train==label,1],
#                c=np.random.rand(3,), alpha=OPACITY,
#                label="Class: {}".format(i))
#
#plt.legend()

pca = PCA()
pcaTrainData = pca.fit_transform(X_train)

plt.figure()
plt.title("Data Set With PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
for i, label in enumerate(distincttrainLabels):
    _ = np.random.rand() # This avoids the strange thing the random number
    # generator does while generating new colors for scatter plot
    plt.scatter(pcaTrainData[y_train==label,0], pcaTrainData[y_train==label,1],
                c=np.random.rand(3,), alpha=OPACITY,
                label="Class: {}".format(i))

plt.legend()
