import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from DataGenerator import generateData
from Preprocessing import deleteFeaturesRandomly
from Filtering import fisherScoreFilter
from settings import (NUMBER_OF_CLASSES, NUMBER_OF_FEATURES, NUMBER_OF_RECORDS_PER_CLASS,
                      FEATURE_MEAN_RANGE, RANDOM_NUMBER_SEED, NUMBER_OF_FEATURES_TO_PRUNE,
                      TEST_SIZE_PERCENTAGE, NOISE_MEAN, NOISE_STD, constantFilterThreshold,
                      correlationFilterThreshold, maxNumberOfFeaturesToRemove)

data, labels = generateData(NUMBER_OF_CLASSES, NUMBER_OF_FEATURES,
                            NUMBER_OF_RECORDS_PER_CLASS, FEATURE_MEAN_RANGE,
                            RANDOM_NUMBER_SEED)
    
prunedTrainData = deleteFeaturesRandomly(data, labels, NUMBER_OF_FEATURES_TO_PRUNE,
                                         randomNumberSeed=RANDOM_NUMBER_SEED)
    
X_train, X_test, y_train, y_test = train_test_split(prunedTrainData, labels,
                                                    test_size=TEST_SIZE_PERCENTAGE)
accuracy = np.zeros(maxNumberOfFeaturesToRemove+1)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
     
accuracy[0] = accuracy_score(y_test, y_pred)

for i in range(1, maxNumberOfFeaturesToRemove+1):
    print(i)
    X_train_i, fisherScores, removedFeatures = fisherScoreFilter(i, X_train, y_train, NUMBER_OF_CLASSES)

    classifier = KNeighborsClassifier(n_neighbors=5)  
    classifier.fit(X_train_i, y_train)

    X_test_i = np.delete(X_test, np.s_[removedFeatures], axis=1)
    y_pred = classifier.predict(X_test_i)

    accuracy[i] = accuracy_score(y_test, y_pred)

plt.figure()
plt.plot(accuracy)
plt.xlabel('Number of features removed')
plt.ylabel('Accuracy')
plt.grid()
plt.show()
#print(confusion_matrix(y_test, y_pred))  
#print(classification_report(y_test, y_pred))  
