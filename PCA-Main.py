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
from Preprocessing import deleteFeaturesRandomly

NUMBER_OF_CLASSES = 4
NUMBER_OF_FEATURES = NUMBER_OF_CLASSES*2
NUMBER_OF_FEATURES_PER_CLASS = 300
TOTAL_NUMBER_OF_RECORDS = NUMBER_OF_CLASSES * NUMBER_OF_FEATURES_PER_CLASS
    
FEATURE_MEAN_RANGE = [0, 50]

RANDOM_NUMBER_SEED = 2
NUMBER_OF_FEATURES_TO_PRUNE = 4

TEST_SIZE_PERCENTAGE = 0.2

np.random.seed(RANDOM_NUMBER_SEED)

data, labels = generateData(NUMBER_OF_CLASSES, NUMBER_OF_FEATURES,
                            NUMBER_OF_FEATURES_PER_CLASS, FEATURE_MEAN_RANGE,
                            RANDOM_NUMBER_SEED)
prunedtrainData = deleteFeaturesRandomly(data, labels, NUMBER_OF_FEATURES_TO_PRUNE, 
                                    randomNumberSeed=RANDOM_NUMBER_SEED)

X_train, X_test, y_train, y_test = train_test_split(prunedtrainData, labels,
                                                    test_size=TEST_SIZE_PERCENTAGE)

distincttrainLabels = np.unique(labels)

plt.figure()
plt.title("Feature Selection With PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
for label in distincttrainLabels:    
    plt.scatter(X_train[y_train==label,0], X_train[y_train==label,1],
                c=np.random.rand(3,))

pca = PCA()
pcaTrainData = pca.fit_transform(X_train)

plt.figure()
plt.title("Feature Selection With PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
for label in distincttrainLabels:    
    plt.scatter(pcaTrainData[y_train==label,0], pcaTrainData[y_train==label,1],
                c=np.random.rand(3,))
    