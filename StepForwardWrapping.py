# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:10:49 2019

@author: ErikF
"""

import numpy as np
import sklearn.neighbors as sk

def StepForwardWrapping(data, labels, nrFeatures, k = 5): 
    nrClasses = len(set(labels))
    nrDataPts, nrFeaturesOriginal = data.shape
   # nrDataPoints = data.shape[0]
    features = []
    for i in range(nrFeatures):
        for j in range(nrFeaturesOriginal):
            sk.KNeighborsClassifier(n_neighbours = k).fit(data,labels)
            

            return data
        
