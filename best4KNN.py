import numpy as np
import pandas as pd
import random
import csv

data = pd.DataFrame(0.00, index = np.arange(1000), columns = ['random', 'X1', 'X2', 'Y']) #establish trade data frame filled with zeroes
#column of numbers 0 through 1000 in random order to be used later to sort data
#points in a random order so that train and test data is spread across the
#dataset but the data points used for train and test are the same for the knn
#and lin reg learners for comparative purposes
random_nums = random.sample(range(0, 1000), 1000)
data['random'] = random_nums
for k in range(0, 1000):
    data['X1'][k] = (k+1) * 2
    data['X2'][k] = (k+1) * 3
#set Y across 2 separate linear functions. The model from the lin reg learner
#will try to approximate both as one linear function, consequently producing
#worse results than those taken from the knn learner, which does not produce a
#model approximating functions
for i in range(0, 500):
    data['Y'][i] = data['X1'][i] #Y = X1 for data points 0 to 500
for j in range(500, 1000):
    data['Y'][j] = -1*data['X2'][j] #Y = -X2 for data points 500 to 1000
data = data.set_index(['random'])
data = data.sort_index()
#print data


b = np.array([data['X1'], data['X2'], data['Y']])
np.savetxt("best4KNN.csv", b.transpose(), delimiter=",")


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = data['X1']
y = data['X2']
z = data['Y']

ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

plt.show()