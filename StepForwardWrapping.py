# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:10:49 2019

@author: ErikF
"""

import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier

def StepForwardWrapping(data, labels, nrFeatures, k = 5): 
    nrClasses = len(set(labels))
    nrDataPts, nrFeaturesOriginal = data.shape
   # nrDataPoints = data.shape[0]
    features = []
    feature_selector = SequentialFeatureSelector(KNeighborsClassifier(5),
               k_features=k,
               forward=True,
               verbose=0,
               cv=5,
               n_jobs=-1)

    features = feature_selector.fit(data, labels)
    return features 
        
