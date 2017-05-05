# -*- coding: utf-8 -*-
"""
Created on Fri May 05 16:18:29 2017

@author: Lima
"""

import numpy as np
import random as rand

def activation(x_i, x,w_i,theta_i,N):
    for j in range(N):
        a = w_i[j] * x[j] + theta_i
    if a >= 0: return 1
    return 0

def E(x,W,theta):
    return -0.5*np.dot(np.dot(x.T,W),x)-np.dot(x.T,theta)
    

def optimize(n):
    W = np.ones([n,n])*-2 - np.identity(n)*-2
    theta = np.ones(n)
    x = np.zeros(n)
    Es = []
    while(E(x,W,theta) >= 0):
        i=rand.randint(0,n-1)
        x[i] = activation(x[i],x, W[i,:], theta[i], n)
    
    
    return x, E(x,W,theta)


#matplotlib inline

import urllib2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.misc as sp

f=urllib2.urlopen("https://upload.wikimedia.org/wikipedia/en/2/24/Lenna.png")

x1 = mpimg.imread(f)
x1 = np.mean(sp.imresize(x1,10),2)
x1[x1 < np.mean(x1.flatten())] = -1    # Black
x1[x1 >= np.mean(x1.flatten())] = 1 # White
x1.astype('int32')

x2 = np.fliplr(x1)

plt.figure()
plt.subplot(121)
imgplot = plt.imshow(x1, cmap='gray')
plt.axis('off')
plt.subplot(122)
imgplot = plt.imshow(x2, cmap='gray')
plt.axis('off')
plt.show()

def hoptrain(x ):
    N = np.size(x[0][0])
    W = np.zeros([N,N])



# train Hopfield net

w = hoptrain([x1,x2])

# corrupt images

n = np.floor(x1.size/2)

cx1 = x1.copy()
p = np.random.permutation(x1.size)
cx1[np.unravel_index(p[:n],x1.shape)] = np.random.randint(0,1,n)

cx2 = x2.copy()
p = np.random.permutation(x2.size)
cx2[np.unravel_index(p[:n],x2.shape)] = np.random.randint(0,1,n)

# test associative memory properties

p1 = hoptest(cx1,w,1)
p2 = hoptest(cx2,w,2)

plt.figure()
plt.subplot(221)
imgplot = plt.imshow(cx1, cmap='gray')
plt.axis('off')
plt.subplot(222)
imgplot = plt.imshow(cx2, cmap='gray')
plt.axis('off')
plt.show()
plt.subplot(221)
imgplot = plt.imshow(p1, cmap='gray')
plt.axis('off')
plt.subplot(222)
imgplot = plt.imshow(p2, cmap='gray')
plt.axis('off')
plt.show()