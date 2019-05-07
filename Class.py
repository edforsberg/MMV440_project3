import numpy as np

class Class():

    def __init__(self, featureMeans, featureCovariances):
        if not isinstance(featureMeans, np.ndarray):
            featureMeans = np.ndarray(featureMeans)

        if not isinstance(featureCovariances, np.ndarray):
            featureCovariances = np.ndarray(featureCovariances)

        if featureMeans.ndim != 1:
            raise Exception("The number of dimensions input argument" +
            "featureMeans must be 1")

        self.numberOfFeatures = len(featureMeans)

        self.featureMeans = featureMeans

        if featureCovariances.ndim != 2:
            raise Exception("The input argument featureCovariances's"+
            "number of dimensions should be 2")

        height, width = featureCovariances.shape
        if height != width:
            raise Exception("The inpurt argument featureCovariances matrix"+
            "must be square")

        if height != self.numberOfFeatures:
            raise Exception("The inpurt argument featureCovariances matrix"+
            "height must be equal to the size of input argument featureMeans")

        self.featureCovariances = featureCovariances
