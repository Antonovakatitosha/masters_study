# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 11:43:05 2022

@author: tijanasustersic
"""

import numpy as np
import matplotlib.pyplot as plt
 
# define Unit Step Function
def unitStep(v):
    if v >= 0: return 1
    else: return 0
  
# design Perceptron Model
def perceptronModel(x, w, b):
    v = np.dot(w, x) + b
    y = unitStep(v)
    return y

# AND Logic Function
# w1 = 1, w2 = 1, b = -1.5
def AND_logicFunction(x):
    w = np.array([1, 1])
    b = -1.5
    return perceptronModel(x, w, b)

# testing the Perceptron Model
print("AND({}, {}) = {}".format(0, 1, AND_logicFunction(np.array([0, 1]))))
print("AND({}, {}) = {}".format(1, 1, AND_logicFunction(np.array([1, 1]))))
print("AND({}, {}) = {}".format(0, 0, AND_logicFunction(np.array([0, 0]))))
print("AND({}, {}) = {}".format(1, 0, AND_logicFunction(np.array([1, 0]))))


# visualisation
area = 200
fig = plt.figure(figsize=(6, 6))
plt.title('The AND Gate', fontsize=20)
ax = fig.add_subplot(111)
# color red: is class 0 and color blue is class 1.
ax.scatter(0, 0, s=area, c='r', label="Class 0")
ax.scatter(0, 1, s=area, c='r', label="Class 0")
ax.scatter(1, 0, s=area, c='r', label="Class 0")
ax.scatter(1, 1, s=area, c='b', label="Class 1")
plt.grid()
plt.show()