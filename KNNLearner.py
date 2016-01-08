"""
knn learner
"""

import numpy as np

class KNNLearner(object):

    def __init__(self, k):
        self.k = k 
    
    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.dataX = dataX
        self.dataY = dataY

        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """

        model_coef = np.zeros(len(points))
        for j in range(0, len(points)):
            distance = np.zeros(len(self.dataX))
            for i in range(0, len(self.dataX)): #establish nearest neighbors according to distance
                for l in range(0, points.shape[1]):
                    distance[i] += (self.dataX[i, l] - points[j, l]) * (self.dataX[i, l] - points[j, l]);
                    sort_indices = np.argsort(distance) #sort according to distance
                    select_var = np.zeros(self.k)
                    for n in range(0, self.k):
                        select_var[n] = self.dataY[sort_indices[n]]
                    model_coef[j] = np.mean(select_var)
        return model_coef


if __name__=="__main__":
    print "the secret clue is 'zzyzx'"