# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:12:10 2017

@author: Lima
"""

import scipy.io
mat = scipy.io.loadmat('digits.mat')
    
X = mat['digits'].astype('float32')

import numpy as np


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
        print(idx)
        prm = np.random.permutation(X.shape[1])
        for i in prm:
            
            # Implementation of the CD-1 algorithm
            ph0 = sigmoid(c + np.dot(W, X[:,i]))
            h_0 = [np.random.rand() < p for p in ph0]
                
            pv1 = sigmoid(b + np.dot(h_0, W))
            
            v_1 = [np.random.rand() < p for p in pv1]
                
            ph1 = sigmoid(c + np.dot(W, v_1))
            
            deltaW = np.outer(ph0, X[:,i]) - np.outer(ph1,v_1)  
            
            deltab = X[:,i] - v_1
            
            deltac = ph0 - ph1
            
            W += 0.01*deltaW
            b += 0.01*deltab
            c += 0.01*deltac
            

    return W,b,c
#W, b, c = RBM_train(X)
import matplotlib.pyplot as plt

for i in range(10):
    plt.imshow(np.reshape(W[i], [28,28]), cmap = 'gray')
    plt.show()

def RBM_test(W, b, c, T):
    v_k = []
    h_k = [np.random.randint(0,2) for i in range(0,10)] 
    
    for idx in range(T):
            
        # Implementation of the CD-K  algorithm
    
        pvk = sigmoid(b + np.dot(h_k, W))
        v_k = [np.random.rand() < p for p in pvk]
                
        
        phk = sigmoid(c + np.dot(W, v_k))
        h_k = [np.random.rand() < p for p in phk]
            

        
        
    return v_k
            
for i in range(10):
    fantasy = RBM_test(W, b, c, 100)
    plt.imshow(np.reshape(fantasy, [28,28]), cmap = 'gray')
    plt.show()
