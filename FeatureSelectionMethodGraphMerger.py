# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:38:52 2019

@author: kaany
"""

import numpy as np
from matplotlib import pyplot as plt

backwardFileName = ""
forwardFileName = ""
filterFileName = ""

backward = np.load(backwardFileName)
forward = np.load(forwardFileName)
_filter = np.load(filterFileName)

nFeaturesToRemoveRange = range(0,11)

plt.figure()
plt.errorbar(nFeaturesToRemoveRange, backward.meanTestAccuracies[nFeaturesToRemoveRange],
             yerr=backward.stdTestAccuracies[nFeaturesToRemoveRange],
             label="Backward Wrapping", capthick=2, capsize=10)
plt.errorbar(nFeaturesToRemoveRange, forward.meanTestAccuracies[nFeaturesToRemoveRange],
             yerr=forward.stdTestAccuracies[nFeaturesToRemoveRange],
             label="Forward Wrapping", capthick=2, capsize=10)
plt.errorbar(nFeaturesToRemoveRange, _filter.meanTestAccuracies[nFeaturesToRemoveRange],
             yerr=_filter.stdTestAccuracies[nFeaturesToRemoveRange],
             label="Filter Wrapping", capthick=2, capsize=10)
plt.title("Accuracy Comparison Of\n Different Feature Selection Methods")
plt.xlabel("Number Of Features Removed")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
