# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 09:46:44 2017

@author: Lima
"""

import numpy as np
import random as rnd
# This data is a representation of the logical OR, where X is the input and Y is the output. 
# The first row of X stands for the bias.
X = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]],dtype='float32').transpose()
T = np.array([0,1,1,1],dtype='float32').reshape([1,4])




def train_slp(X, Y, eta, nepochs):
    K = len(X.T[0])
    N = len(X.T)
    w = rnd.random()*0.1*np.ones(K)
    RMSE = []
    for n in range(0,nepochs):
        error = []
        for i in rnd.sample(range(0, N), N):
            y = np.dot(w.T, X.T[i])
            print(y)
            print(Y.T[i])
            print()
            wdelta = -eta*(y - Y.T[i])*X.T[i]
            error.append( 0.5*sum((y - Y.T[i])**2) )
            w = w + wdelta 
        RMSE.append(np.sqrt(np.mean(error)))

    return w, RMSE
    
w, RMSE = train_slp(X, T, 0.1, 100)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(RMSE)
plt.title('Performance as function of epoch' )
plt.xlabel('Epoch')
plt.ylabel('Root Mean Square Error')
plt.xlim(-5, 100)

def test_performance(Pattern, Targets):
    w, RMSE = train_slp(Pattern, Targets, 0.1, 100)
    fig, ax = plt.subplots()
    ax.plot(RMSE)
    plt.title('Performance as function of epoch' )
    plt.xlabel('Epoch')
    plt.ylabel('Root Mean Square Error')
    plt.xlim(-5, 100)
    
    
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')

x = np.ones(100)
y = np.linspace(-1,1,100)
ax.plot(1,y,-y)
ax.scatter(X[0], X[1], X[2])



