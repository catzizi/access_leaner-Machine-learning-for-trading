"""
knn learner
"""

import numpy as np
import KNNLearner as knn
import random
import pandas as pd
import math

class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost):
        self.bags = bags
        self.learner = learner
        self.kwargs = kwargs

    
    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.dataX = dataX
        self.dataY = dataY

        trX = []
        trY = []
        
        for i in range(0,self.bags):
            m = len(self.dataY)*60 // 100 #ammount of training data randomized in each bag

            B = np.random.randint(len(self.dataX),size=m) #random rows of data to be used in the bag

            trainingX = self.dataX[B,:]
            trainingY = self.dataY[B]

            trX.append(trainingX)
            trY.append(trainingY)
        self.trX = trX
        self.trY = trY
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """

        
        model_coef = np.zeros(len(points))

        learners = []
        
        for k in range(0,self.bags):

            learner = self.learner(**self.kwargs)
            learner.addEvidence(self.trX[k], self.trY[k])
            predY = learner.query(points)
            learners.append(predY)

        model_coef = np.mean(learners, axis = 0) #average of values predicted from each bag
        return model_coef



if __name__=="__main__":
    print "the secret clue is 'zzyzx'"