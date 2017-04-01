# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 09:38:05 2017

@author: Lima
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

X, Y, Z = axes3d.get_test_data(0.1)

# sample training data points
X_train = np.vstack([X[::2,::2].flatten(), Y[::2,::2].flatten()])
T_train = Z[::2,::2].flatten()

# sample test data points
X_test = np.vstack([X[1::2,1::2].flatten(), Y[1::2,1::2].flatten()])
T_test = Z[1::2,1::2].flatten()

import numpy as np
from scipy.spatial.distance import pdist
from scipy.linalg import pinv
import random

def rbf_train(K, X, T):
    # input:
    # M x N training samples X
    # 1 x N target outputs T
    #
    # output:
    # K x M prototype vectors mu
    # 1 x 1 scalar Gaussian with sigma
    # K x 1 output weights w

    [M,N] = X.shape
    

    # select K datapoints at random

    mu = random.sample(X.T, K)

    # compute global distance parameter
    
    d_max = max(pdist(mu))
    sigma = d_max/(np.sqrt(2*K))
        
    # compute w
    
    
    return [mu, sigma, w]