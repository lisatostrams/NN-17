# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 10:12:12 2017

@author: Lima
"""

import scipy.stats as ss
import numpy as np

# read data from mat file
import scipy.io
mat = scipy.io.loadmat('MLP_data.mat')
    
X_train = mat['X_training']
X_test = mat['X_test']
T_train = mat['T_training']
T_test = mat['T_test']

# take out all constant pixels since they are noninformative anyway
idx = (np.std(X_train,1) != 0) & (np.std(X_test,1) != 0)
X_train = X_train[idx,:]
X_test = X_test[idx,:]

# zscore data
X_train = np.array(ss.zscore(X_train,1))
X_test = np.array(ss.zscore(X_test,1))

# add bias terms
X_train = np.vstack([np.ones([1,X_train.shape[1]]),X_train])
X_test = np.vstack([np.ones([1,X_test.shape[1]]),X_test])

import numpy as np
import scipy.stats as ss

def sigmoid(x):
    """
    Sigmoid function; returns function value and gradient
    """

    fx = 1.0 / (1 + np.exp(-x))
    gradx = fx * (1 - fx)

    return fx, gradx

def error(f_a_3,T):
    """
    Computes squared error divided by number of trials
    
    Input:
    f_a_3 : MLP output states
    T   : noutput x ntrials targets

    Output:
    E_w        : squared error
    
    """
   
    ntrials = T.shape[1]

    E_w = 1.0 / (2 * ntrials) * np.sum(np.sum((f_a_3 - T) ** 2))
   
    return E_w

def forwardprop(W_1, W_2, X):
    """
    Performs forward propagation
    
    Input:
    W_1 : nhidden x ninput input-to-hidden weight matrix
    W_2 : noutput x nhidden hidden-to-output weight matrix
    X   : ninput x ntrials input data
    
    Output:
    f_a_2 : MLP hidden unit states
    f_a_3 : MLP output states
    grad_f_a_2 : gradient of the hidden unit activation function
    grad_f_a_3 : gradient of the output unit activation function
    """
    
    # You should now implement the forward propagation function. Your
    # implementation should compute and return the outputs of the second and
    # third layer units as well as their gradients.

    # First, compute the inputs of the second layer units (i.e. a_2). Write
    # your code below:
    # -------------------------------------------------------------------------
    a_2 = np.dot(W_1, X)
    # -------------------------------------------------------------------------

    # Once you have computed a_2, use it with the sigmoid function that you
    # have implemented (i.e. sigmoid) to compute the outputs of the second
    # layer units (i.e. f_a_2) and their gradients (i.e. grad_f_a_2). Write
    # your code below:
    # -------------------------------------------------------------------------
    f_a_2, grad_f_a_2 = sigmoid(a_2) 
    # -------------------------------------------------------------------------

    # Then, compute the inputs of the third layer units (i.e. a_3). Write your
    # code below:
    # -------------------------------------------------------------------------

    a_3 = np.dot(W_2, f_a_2)
    
    # -------------------------------------------------------------------------

    # Once you have computed a_3, use it with the sigmoid function that you
    # have implemented (i.e. sigmoid) to compute the outputs of the third layer
    # units (i.e. f_a_3) and their gradients (i.e. grad_f_a_3). Write your code
    # below:
    # -------------------------------------------------------------------------

    f_a_3, grad_f_a_3 = sigmoid(a_3)
    
    # -------------------------------------------------------------------------

    return f_a_2, f_a_3, grad_f_a_2, grad_f_a_3

def backprop(f_a_2, f_a_3, grad_f_a_2, grad_f_a_3, T, W_2, X):
    """
    Performs backpropagation step
    
    Input:
    f_a_2 : MLP hidden unit states
    f_a_3 : MLP output states
    grad_f_a_2 : gradient of the hidden unit activation function
    grad_f_a_3 : gradient of the output unit activation function
    T   : noutput x ntrials targets
    W_2 : noutput x nhidden hidden-to-output weight matrix
    X   : ninput x ntrials input data
    
    Output:
    grad_E_w_1 : ntrials x 1 gradient of the error w.r.t W_1
    grad_E_w_2 : ntrials x 1 gradient of the error w.r.t W_2
    """
        
    # You should now implement the back propagation function. Your
    # implementation should compute and return the gradients of the error
    # function.

    # First, compute the errors of the second and third layer units (i.e.
    # delta_2 and delta_3). Write you code below:
    # -------------------------------------------------------------------------
    

    delta_3 = (f_a_3 - T) * grad_f_a_3
                   
    delta_2 = grad_f_a_2 *(np.dot(W_2.T, delta_3))
    
    # -------------------------------------------------------------------------

    # Once you have computed delta_2 and delta_3, use them to compute the
    # gradients of the error function (i.e. grad_E_w_1 and grad_E_w_2). Write
    # your code below:
    # -------------------------------------------------------------------------
 
    # Add your solution here.
    grad_E_w_2 = np.dot(delta_3, f_a_2.T)
    
    grad_E_w_1 = np.dot(delta_2, X.T)
    # -------------------------------------------------------------------------
    
    return grad_E_w_1, grad_E_w_2

nepochs = 2000
learning_rate = 0.001

ninput = X_train.shape[0]
noutput = T_train.shape[0]
nhidden = 15

# initialize weights
r = np.sqrt(6)/np.sqrt(nhidden+ninput)
W_1 = np.random.uniform(-r, r, [nhidden,ninput])

r = np.sqrt(6)/np.sqrt(nhidden+ninput)
W_2 = np.random.uniform(-r, r, [noutput,nhidden])

# keep track of errors
train_error = np.zeros([nepochs+1,1])
test_error = np.zeros([nepochs+1,1])

# training
for epoch in xrange(0,nepochs):

    # First, use the forward propagation function that you have implemented
    # (i.e. forwardprop) to compute the outputs of the second and third layer
    # units (i.e. f_a_2 and f_a_3) as well as their gradients (i.e. grad_f_a_2
    # and grad_f_a_3). Write your code below:
    # -------------------------------------------------------------------------
    [f_a_2, f_a_3, grad_f_a_2, grad_f_a_3] = forwardprop(W_1, W_2, X_train)
    # -------------------------------------------------------------------------

    # compute error
    train_error[epoch] = error(f_a_3, T_train)
    test_error[epoch] = error(forwardprop(W_1, W_2, X_test)[1], T_test)

    if (epoch + 1) % 100 == 0:
         print('Iteration: ' + str(epoch+1) + ' / ' + str(nepochs) + '; Train error: ' 
               + str(train_error[epoch])) + '; Test error: ' + str(test_error[epoch])
 
    # Once you have computed f_a_2, f_a_3, grad_f_a_2 and grad_f_a_3, use them
    # with the back propagation function that you have implemented (i.e.
    # backprop) to compute the gradients of the error function (i.e. grad_E_w_1
    # and grad_E_w_2). Write your code below:
    # -------------------------------------------------------------------------
    [grad_E_w_1, grad_E_w_2] = backprop(f_a_2, f_a_3, grad_f_a_2, grad_f_a_3, T_train, W_2, X_train)
    # -------------------------------------------------------------------------
             
    W_1 = W_1 - learning_rate * grad_E_w_1                                 
    W_2 = W_2 - learning_rate * grad_E_w_2                                                                            
    
# get error after the last update
train_error[-1] = error(forwardprop(W_1, W_2, X_train)[1], T_train)
test_error[-1] = error(forwardprop(W_1, W_2, X_test)[1], T_test)

import matplotlib.pyplot as plt