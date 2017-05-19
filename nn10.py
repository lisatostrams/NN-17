# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:12:10 2017

@author: Lima
"""

import scipy.io
mat = scipy.io.loadmat('digits.mat')
    
X = mat['digits'].astype('float32')

import numpy as np

def gibbs(x, W, b, ngibbs=20, temp=1.0):

    M = x.size # number of variables
    
    # perform Gibbs sampling
    XM = np.zeros([ngibbs,M])
    XM[0] = x
    for t in range(1,ngibbs):
        for i in range(M):
            pi = sigmoid((np.dot(W[i,:],x) + b[i])/temp)
            x[i] = np.random.rand() < pi
        XM[t,:] = x
    
    return XM

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def RBM_train(X,N=10,T=10):
    # RBM_TRAIN trains a restricted Boltzmann machine on M x nexamples input
    # data X with N hidden units for T time steps.
    #
    # Returns N x M weights W, M x 1 visible unit bias b and N x 1 hidden unit bias c

    M = X.shape[0]
    W = 10**-1*np.random.normal(size=[N,M])
    b = 10**-1*np.random.normal(size=M)
    c = np.zeros(N)
    
    # iterate over examples (iterating over minibatches would be more elegant!)
    for idx in range(T):

        prm = np.random.permutation(X.shape[1])
        for i in prm:
            
            # Implementation of the CD-1 algorithm
            gibbs(X, W, b)
            

    return W,b,c

W, b, c = RBM_train(X)