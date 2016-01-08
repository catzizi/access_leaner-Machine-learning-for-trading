# access_leaner-Machine-learning-for-trading
implement and evaluate three learning algorithms as Python classes: A KNN learner, a Linear Regression learner (provided) and a Bootstrap Aggregating learner
The classes have been named KNNLearner, LinRegLearner, and BagLearner respectively.  
Considering this a regression problem (not classification). So the goal is to return a continuous numerical result (not a discrete result).

In this project I am training & testing with static spatial data.

KNNLearner class should be implemented in the file KNNLearner.py
Example:
import KNNLearner as knn
learner = knn.KNNLearner(k = 3) # constructor
learner.addEvidence(Xtrain, Ytrain) # training step
Y = learner.query(Xtest) # query
Where "k" is the number of nearest neighbors to find.
Xtrain and Xtest should be ndarrays (numpy objects)
where each row represents an X1, X2, X3... XN set of feature values. 
The columns are the features and the rows are the individual example instances. Y and Ytrain are single dimension ndarrays that indicate the value
we are attempting to predict with X.
Implement BagLearner 
Implement Bootstrap Aggregating as a Python class named BagLearner.
BagLearner class should be implemented in the file BagLearner.py.
example:
import BagLearner as bl
learner = bl.BagLearner(learner = knn.KNNLearner, kwargs = {"k":3}, bags = 20, boost = False)
learner.addEvidence(Xtrain, Ytrain)
Y = learner.query(Xtest)
Where learner is the learning class to use with bagging. kwargs are keyword arguments to be passed on to the learner's constructor and they vary according to the learner (see hints below). "bags" is the number of learners you should train using Bootstrap Aggregation. If boost is true, then you should implement boosting.
