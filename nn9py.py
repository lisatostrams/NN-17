# -*- coding: utf-8 -*-
"""
Created on Fri May 12 10:35:24 2017

@author: Lima
"""

import numpy as np
import urllib2
import matplotlib.image as mpimg
import scipy.misc as sp

f=urllib2.urlopen("https://upload.wikimedia.org/wikipedia/en/2/24/Lenna.png")

x1 = mpimg.imread(f)
x1 = np.mean(sp.imresize(x1,10),2)
imsize = x1.shape
x1[x1 < np.mean(x1.flatten())] = -1    # Black
x1[x1 >= np.mean(x1.flatten())] = 1 # White
x1.astype('int32')

x2 = np.fliplr(x1)

import math
import random as rand

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def bmtrain(x, niter=10, temp=1.0, ngibbs=20, burnin=10, eta=0.01):
    """
    x: list of examples
    niter: number of gradient steps
    temp: temperature parameter
    ngibbs: number of Gibbs sampling steps
    burnin: burn-in
    eta: learning rate
    """

    N = len(x) # number of examples
    M = x[0].size # number of variables

    # weights of Boltzmann machine
    W = np.zeros([M,M])
    
    # biases of Boltzmann machine
    b = np.zeros(M)

    # flatten images
    for j in range(N):
        x[j] = x[j].reshape([M,1])
        
    # create m x n design matrix
    X = np.array(x).squeeze()
    
    # cast values to [0,1]
    u, indices = np.unique(X, return_inverse=True)
    assert len(u)==2
    X[:] = indices.reshape(X.shape)
            
    for it in range(niter):
        
        print "iteration {0} of {1}".format(it+1,niter)
        
        # compute expectations under the empirical distribution
    
        dE_dW = np.dot(-X.T, X)
        EP_w = sum(dE_dW) / N

        dE_db = np.mean(-X, axis=0)
        EP_b = sum(dE_db )/N
        
        # FILL IN DETAILS HERE; Use dE_dW and dE_db to denote the gradients
        
        # perform Gibbs sampling
        
        rand_i = rand.randint(0,N-1)
        Xm = bmtest(X[rand_i], W, b)
        Xm = Xm[burnin:]
        # FILL IN DETAILS HERE

        # compute expectations under the model distribution
        
        dEM_dW = np.dot(-Xm.T, Xm)
        EpM_w = sum(dEM_dW) / ngibbs
                   
        dEM_db = np.mean(-Xm, axis=0)
        EpM_b = sum(dEM_db) / ngibbs
        # FILL IN DETAILS HERE; Use dEM_dW and dEM_db to denote the gradients
        
        
        
        # update parameters
        W += eta * (dEM_dW - dE_dW)
        b += eta * (dEM_db - dE_db)
    
    # force symmetry
    W = (W + W.transpose()) / 2
    
    return W, b

def bmtest(x, W, b, ngibbs=20, temp=1.0):

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


T = 1.0

# train Boltzmann machine
W,b = bmtrain([x1, x2],niter=3, temp=T)

# test Boltzmann machine
x = np.random.randint(0,2,x1.size).astype('float')
XM = bmtest(x, W, b, temp=T)
import matplotlib.pyplot as plt
plt.figure()
plt.subplot(121)
imgplot = plt.imshow(x1, cmap='gray')
plt.axis('off')
plt.subplot(122)
imgplot = plt.imshow(x2, cmap='gray')
plt.axis('off')
plt.show()

for j in range(1, len(XM)):
    XM1 = np.reshape(XM[j-1], (51,51))
    plt.subplot(4,5,j)
    imgplot = plt.imshow(XM1, cmap='gray')
    plt.axis('off')
plt.show()
    