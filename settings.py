# -*- coding: utf-8 -*-
"""
Created on Thu May  9 19:22:57 2019

@author: kaany
"""

NUMBER_OF_CLASSES = 6
NUMBER_OF_FEATURES = NUMBER_OF_CLASSES*2
NUMBER_OF_RECORDS_PER_CLASS = 500

FEATURE_MEAN_RANGE = [0, 10]

RANDOM_NUMBER_SEED = 0
NUMBER_OF_FEATURES_TO_PRUNE = int(NUMBER_OF_FEATURES/2)

TEST_SIZE_PERCENTAGE = 0.2

NOISE_MEAN = 10
NOISE_STD = 5

constantFilterThreshold = 10 
correlationFilterThreshold = 1
numberOfFeaturesToRemove = 2
