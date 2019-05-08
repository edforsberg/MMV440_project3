# -*- coding: utf-8 -*-
"""
Created on Tue May  7 18:07:06 2019

@author: kaany
"""
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score  
from mlxtend.feature_selection import SequentialFeatureSelector

from DataGenerator import generateData
from Preprocessing import deleteFeaturesRandomly

NUMBER_OF_CLASSES = 6
NUMBER_OF_FEATURES = NUMBER_OF_CLASSES*2
NUMBER_OF_FEATURES_PER_CLASS = 500
TOTAL_NUMBER_OF_RECORDS = NUMBER_OF_CLASSES * NUMBER_OF_FEATURES_PER_CLASS

FEATURE_MEAN_RANGE = [0, 10]

RANDOM_NUMBER_SEED = 3
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

n_neighbors = 5
nFeaturesToSelect = 4

feature_selector = SequentialFeatureSelector(KNeighborsClassifier(n_neighbors),  
           k_features=nFeaturesToSelect,
           forward=False,
           verbose=2,
           cv=5,
           n_jobs=-1)

features = feature_selector.fit(X_train, y_train)
print(features.k_feature_idx_)
xTrainWithSelectedFeatures = X_train[:, features.k_feature_idx_]
xTestWithSelectedFeatures = X_test[:, features.k_feature_idx_]

knn = KNeighborsClassifier(n_neighbors)
knn.fit(xTrainWithSelectedFeatures, y_train)

train_pred = knn.predict(xTrainWithSelectedFeatures)  
accuracyTrain = accuracy_score(y_train, train_pred)
print('Accuracy on training set: {}'.format(accuracyTrain))

test_pred = knn.predict(xTestWithSelectedFeatures)   
accuracyTest = accuracy_score(y_test, test_pred)
print('Accuracy on test set: {}'.format(accuracyTest)) 


