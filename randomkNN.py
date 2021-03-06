import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from DataGenerator import generateData
from Preprocessing import transfromFeaturesToNoiseRandomly
from Filtering import fisherScoreFilter
from settings import (NUMBER_OF_CLASSES, NUMBER_OF_FEATURES, NUMBER_OF_RECORDS_PER_CLASS,
                      FEATURE_MEAN_RANGE, RANDOM_NUMBER_SEED, NUMBER_OF_FEATURES_TO_PRUNE,
                      TEST_SIZE_PERCENTAGE, NOISE_MEAN, NOISE_STD, constantFilterThreshold,
                      correlationFilterThreshold, maxNumberOfFeaturesToRemove)

nRuns = 20
randomSeeds = range(0, nRuns)
accuracy = np.zeros((maxNumberOfFeaturesToRemove, nRuns))
k = int(sys.argv[1])

for r in randomSeeds:
    data, labels = generateData(NUMBER_OF_CLASSES, NUMBER_OF_FEATURES,
                                NUMBER_OF_RECORDS_PER_CLASS, FEATURE_MEAN_RANGE, r)
    
    prunedTrainData = transfromFeaturesToNoiseRandomly(data, labels, NUMBER_OF_FEATURES_TO_PRUNE, NOISE_MEAN, NOISE_STD, r)
    
    X_train, X_test, y_train, y_test = train_test_split(prunedTrainData, labels,
                                                    test_size=TEST_SIZE_PERCENTAGE)

    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
     
    accuracy[0, r] = accuracy_score(y_test, y_pred)

    for i in range(1, maxNumberOfFeaturesToRemove):
        removedFeatures = np.random.choice(12, i, replace=False)
        X_train_i = np.delete(X_train, np.s_[removedFeatures], axis=1)

        classifier = KNeighborsClassifier(n_neighbors=k)  
        classifier.fit(X_train_i, y_train)

        X_test_i = np.delete(X_test, np.s_[removedFeatures], axis=1)
        y_pred = classifier.predict(X_test_i)

        accuracy[i, r] = accuracy_score(y_test, y_pred)

averageAccuracy = np.mean(accuracy, axis=1)

plt.figure()
plt.plot(averageAccuracy)
plt.xlabel('Number of features removed')
plt.ylabel('Accuracy')
plt.title('Random filtering of features')
plt.legend(['Average accuracy of kNN (k = 5, number of runs = 20)'])
plt.grid()
plt.show()


