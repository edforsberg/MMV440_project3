# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:26:49 2019

@author: kaany
"""
import numpy as np


def deleteFeaturesRandomly(data, labels, nFeaturesToDelete,
                           randomNumberSeed=None):

    if not isinstance(data, np.ndarray):
        data = np.ndarray(data)

    nRecords, nFeatures = data.shape

    if nFeaturesToDelete > nFeatures:
        raise Exception("Number of features to prune cannot be greater than"+
                        " the number of features available in the data set.")

    if nFeaturesToDelete < 1:
        return data

    if randomNumberSeed != None:
        np.random.seed(randomNumberSeed)

    distinctClasses = np.unique(labels)
    prunedData = data.copy()

    for classLabel in distinctClasses:
        featureIndicesToZero = np.random.choice(list(range(nFeatures)),
                                                nFeaturesToDelete,
                                                replace=False)
        prunedData[labels==classLabel, featureIndicesToZero[:, np.newaxis]] = 0

    return prunedData

def transfromFeaturesToNoiseRandomly(data, labels, nFeaturesToNoise,
                                     mean, std, randomNumberSeed=None):

    if not isinstance(data, np.ndarray):
        data = np.ndarray(data)

    nRecords, nFeatures = data.shape

    if nFeaturesToNoise > nFeatures:
        raise Exception("Number of features to prune cannot be greater than"+
                        " the number of features available in the data set.")

    if nFeaturesToNoise < 1:
        return data

    if randomNumberSeed != None:
        np.random.seed(randomNumberSeed)

    distinctClasses = np.unique(labels)
    noisyData = data.copy()

    for classLabel in distinctClasses:
        featureIndicesToZero = np.random.choice(list(range(nFeatures)),
                                                nFeaturesToNoise,
                                                replace=False)
        nRecordsForCurrentClass = np.count_nonzero(labels==classLabel)
        noise = np.random.normal(mean, std, size=(nRecordsForCurrentClass,nFeaturesToNoise))
        noisyData[labels==classLabel, featureIndicesToZero[:, np.newaxis]] = noise.transpose()

    return noisyData

def addNoisyData(data, nNoisyFeatures, _range=None):

    if not isinstance(_range, np.ndarray):
        _range = np.ndarray(_range)

    if _range.shape != 2:
        raise Exception("Input argument _range should have 2 elements")

    nRecords = len(data)
    noisyFeatures = np.random.rand(nRecords, nNoisyFeatures)
    if _range != None:
        noisyFeatures = noisyFeatures*max(_range) - min(_range)

    noisyData = np.concatenate((data, noisyFeatures), axis=1)
    return noisyData

