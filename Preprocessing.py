# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:26:49 2019

@author: kaany
"""
import numpy as np


def deleteFeaturesRandomly(data, numberOfClasses):
    pass

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
    
    