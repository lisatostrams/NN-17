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
    for n in range(0,nepochs):
        for i in rnd.sample(range(0, N), N):
            y = np.dot(w.T, X.T[i])
            print(y>0.1)
            print(Y.T[i])
            print()
            wdelta = -eta*(y - Y.T[i])*X.T[i]
            w = w + wdelta    

    return w
    