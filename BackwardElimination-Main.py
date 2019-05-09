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
from matplotlib import pyplot as plt

from DataGenerator import generateData
from Preprocessing import deleteFeaturesRandomly, transfromFeaturesToNoiseRandomly
from Visualization import plotConfusionMatrix

NUMBER_OF_CLASSES = 6
NUMBER_OF_FEATURES = NUMBER_OF_CLASSES*2
NUMBER_OF_FEATURES_PER_CLASS = 500

FEATURE_MEAN_RANGE = [0, 10]

RANDOM_NUMBER_SEED = 0
NUMBER_OF_FEATURES_TO_PRUNE = int(NUMBER_OF_FEATURES / 2)

NOISE_MEAN = 10
NOISE_STD = 5

TEST_SIZE_PERCENTAGE = 0.2

OPACITY = 0.7

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

distinctTrainLabels = np.unique(labels)

n_neighbors = 5
nFeaturesToSelect = 2

feature_selector = SequentialFeatureSelector(KNeighborsClassifier(n_neighbors),
           k_features=max(2,nFeaturesToSelect),
           forward=False,
           verbose=0,
           cv=5,
           n_jobs=-1)

features = feature_selector.fit(X_train, y_train)
print("\nSelected features' indices: {}".format(features.k_feature_idx_))

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

plt.figure()
plt.title("Feature Selection With Backward Selection\n" +
          "({} Features Selected)".format(nFeaturesToSelect))
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
for i, label in enumerate(distinctTrainLabels):
    _ = np.random.rand() # This avoids the strange thing the random number
    # generator does while generating new colors for scatter plot
    plt.scatter(xTrainWithSelectedFeatures[y_train == label, 0],
                xTrainWithSelectedFeatures[y_train == label, 1],
                c=np.random.rand(3,), label="Class " + str(i),
                alpha=OPACITY)
plt.legend()

plotConfusionMatrix(y_test, test_pred, title='Confusion matrix')

