import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from DataGenerator import generateData
from Preprocessing import deleteFeaturesRandomly


NUMBER_OF_CLASSES = 4
NUMBER_OF_FEATURES = NUMBER_OF_CLASSES * 2
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

prunedTrainData = deleteFeaturesRandomly(data, labels, NUMBER_OF_FEATURES_TO_PRUNE,
                                         randomNumberSeed=RANDOM_NUMBER_SEED)

X_train, X_test, y_train, y_test = train_test_split(prunedTrainData, labels,
                                                    test_size=TEST_SIZE_PERCENTAGE)

#Constant filter
constantFilter = VarianceThreshold(threshold=0)
X_train = constantFilter.fit_transform(X_train)

#Quasi-constant filter
quasiConstantFilter = VarianceThreshold(threshold=0.5)
X_train = quasiConstantFilter.fit_transform(X_train)

#Correlation filter
dataFrame = pd.DataFrame(X_train)
correlationMatrix = dataFrame.corr().abs()
upper = correlationMatrix.where(np.triu(np.ones(correlationMatrix.shape), k=1).astype(np.bool))
featuresToDrop = [column for column in upper.columns if any(upper[column] > 0.95)]
dataFrame.drop(dataFrame.columns[featuresToDrop], axis=1)

X_train = np.array(dataFrame)







