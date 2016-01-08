import numpy as np
import pandas as pd
import random
from random import randint
import csv

data = pd.DataFrame(0.00, index = np.arange(1000), columns = ['random', 'X1', 'X2', 'Y']) #establish trade data frame filled with zeroes
#column of numbers 0 through 1000 in random order to be used later to sort data
#points in a random order so that train and test data is spread across the
#dataset but the data points used for train and test are the same for the knn
#and lin reg learners for comparative purposes
random_nums = random.sample(range(0, 1000), 1000)
data['random'] = random_nums
for k in range(0, 250):
    data['X1'][k] = (k+1) * 5 #spacing between data points on X1 axis changes
    data['X2'][k] = data['X1'][k]*data['X1'][k]*data['X1'][k] * 3 #X2 is 3*X1^3
for j in range(250, 500):
    data['X1'][j] = (j+1) * 10 #spacing between data points on X1 axis changes
    data['X2'][j] = data['X1'][j]*data['X1'][j]*data['X1'][j] * 3 
for l in range(500, 750):
    data['X1'][l] = (l+1) * 20
    data['X2'][l] = data['X1'][l]*data['X1'][l]*data['X1'][l] * 3 
for m in range(750, 1000):
    data['X1'][m] = (m+1) * 30 
    data['X2'][m] = data['X1'][m]*data['X1'][m]*data['X1'][m] * 3 
#introduce a good deal of error in the X2 parameter, which is perpendicular
#to the linear relationship between X1 and Y
for n in range(1, 999): 
    v =  randint(1,999)
    data['X2'][v] = data['X2'][v]+data['X2'][v]*randint(-5, 5)
#Y is a linear function of X1, Y = 5*X1
for i in range(0, 1000):
    data['Y'][i] = 5*data['X1'][i]+randint(-100,100)
#introduce some more random error on random points in the Y parameter
for l in range(1, 999):
    u = randint(1,999)
    data['Y'][u] = data['Y'][u]+randint(-100, 100) 
data = data.set_index(['random'])
data = data.sort_index()
#print data



b = np.array([data['X1'], data['X2'], data['Y']])
np.savetxt("best4linreg.csv", b.transpose(), delimiter=",")


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