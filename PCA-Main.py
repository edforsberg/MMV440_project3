# -*- coding: utf-8 -*-
"""
Created on Tue May  7 18:07:06 2019

@author: kaany
"""
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np

from DataGenerator import generateData
from Preprocessing import deleteFeaturesRandomly

NUMBER_OF_CLASSES = 4
NUMBER_OF_FEATURES = NUMBER_OF_CLASSES*2
NUMBER_OF_FEATURES_PER_CLASS = 200
TOTAL_NUMBER_OF_RECORDS = NUMBER_OF_CLASSES * NUMBER_OF_FEATURES_PER_CLASS
    
FEATURE_MEAN_RANGE = [0, 50]
    
RANDOM_NUMBER_SEED = 0
NUMBER_OF_FEATURES_TO_PRUNE = 4

data, labels = generateData(NUMBER_OF_CLASSES, NUMBER_OF_FEATURES,
                            NUMBER_OF_FEATURES_PER_CLASS, FEATURE_MEAN_RANGE,
                            RANDOM_NUMBER_SEED)
prunedData = deleteFeaturesRandomly(data, labels, NUMBER_OF_FEATURES_TO_PRUNE, 
                                    randomNumberSeed=RANDOM_NUMBER_SEED)

distinctLabels = np.unique(labels)

pca = PCA()
pcaData = pca.fit_transform(prunedData)

plt.figure()
plt.title("Feature Selection With PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
for label in distinctLabels:    
    plt.scatter(pcaData[labels==label,0], pcaData[labels==label,1],
                c=np.random.rand(3,))