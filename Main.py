# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:26:22 2019

@author: kaany
"""

from DataGenerator import generateData
from Preprocessing import deleteFeaturesRandomly

RANDOM_NUMBER_SEED = 0
NUMBER_OF_FEATURES_TO_PRUNE = 4

data, labels = generateData(RANDOM_NUMBER_SEED)
prunedData = deleteFeaturesRandomly(data, labels, NUMBER_OF_FEATURES_TO_PRUNE, 
                                    randomNumberSeed=RANDOM_NUMBER_SEED)
