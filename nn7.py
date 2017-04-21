import numpy as np
#% matplotlib inline
import matplotlib.pyplot as plt


def display_2D_coordinates(dims,weights):
  
    weights = weights - np.min(weights[:])
    weights = weights / np.max(weights[:])
  
    w1 = np.reshape(weights[0],dims)
    w2 = np.reshape(weights[1],dims) 
  
    if dims.size==2:
        plt.figure()
        plt.plot(w1,w2,'k')
        plt.plot(w1.transpose(),w2.transpose(),'k') 
        plt.hold(True)
        plt.axis('equal')
        plt.axis([0, 1, 0, 1])
        plt.show()
    else:
        raise Exception('display function does not work on higher-dimensional output')
  
    plt.show()

def som(dataset, dims=np.array([30,30]), niter=50, epsilon=0.02, sigma=0.3, dispfun=display_2D_coordinates, dstep=None):
# SOM self-organizing map algorithm; data is normalized between 0 and 1
#
# input:
# dataset = N x M array of N x 1 input patterns
# dims = dimensionality of the network layer (e.g. [5 5 5])
# niter = number of steps
# epsilon = learning rate
# sigma = standard deviation of the Gaussian neighbourhood function
# dispfun = display function to show online output
# dstep = display after each dstep iterations; defaults to last iteration
#
# output:
# weights = SOM weights

    N = dataset.shape[0]
    M = dataset.shape[1]
    K = np.prod(dims)
    D = dims.size

    if dstep==None:
        dstep = niter
    
    weights = 0.5 - 0.25 * (2*np.random.rand(N,K)-1)

    # create coordinates
    x = np.unravel_index(np.arange(K), dims)
    coords = np.zeros([D,K])
    for d in range(D):
        coords[d] = x[d]
    coords /= np.max(coords[:])
  
    # before training
    dispfun(dims,weights)

    for t in range(niter):
    
        for j in range(M):
            weights = som_update(dataset[:,j],weights,coords,epsilon,sigma)

        # display output
        if not dispfun is None and np.mod(t+1,dstep)==0:
            dispfun(dims,weights)
        
    return weights, coords

def som_update(pattern, weights, coords, epsilon, sigma):
    
    tmp = np.reshape(pattern, (pattern.size, 1)) - weights
                   
    winner = coords[:,np.argmin( np.sum(tmp**2, axis=0) )]
    
    a = np.exp(  (-(coords[0] - winner[0])**2 - (coords[1] - winner[1])**2)/ (2*sigma**2) )
    
    w_delta = epsilon * a * tmp
    
    return weights + w_delta


data = np.random.rand(2,100)
som(data)

#som(data, epsilon=0.01, sigma = 0.3)

#som(data, epsilon=0.01, sigma = 0.5)

som(data, epsilon=0.05, sigma = 0.2)

som(data, epsilon=0.1, sigma = 0.1)
