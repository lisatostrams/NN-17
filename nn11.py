# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:18:50 2017

@author: Lima
"""

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

train, test = datasets.get_mnist()

Xtrain = train._datasets[0].reshape([train._datasets[0].shape[0],1,28,28])
ytrain = train._datasets[1]
Xtest = test._datasets[0].reshape([test._datasets[0].shape[0],1,28,28])
ytest = test._datasets[1]

train = datasets.TupleDataset(Xtrain,ytrain)
test = datasets.TupleDataset(Xtest,ytest)

train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

class MLP(Chain):
     def __init__(self, n_units, n_out):
         super(MLP, self).__init__(
             # the size of the inputs to each layer will be inferred
             l1=L.Convolution2D(1, n_units, ksize=1, stride=3, pad=0),  # n_in -> n_units
             l2=L.Linear(None, n_units),  # n_units -> n_units
             l3=L.Linear(None, n_out),    # n_units -> n_out
         )

     def __call__(self, x):
         h1 = F.relu(self.l1(x))
         h2 = F.relu(self.l2(h1))
         y = self.l3(h2)
         return y
     
model = L.Classifier(MLP(50, 5))  # the input size, 784, is inferred
optimizer = optimizers.SGD()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (20, 'epoch'), out='result')

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
print('Training')
trainer.run()  


import json
import matplotlib.pyplot as plt
#%matplotlib inline

out_dir = '/Users/Lima/Desktop/NN-17/result'

with open(out_dir + '/log') as data_file:
    data = json.load(data_file)

# extract training and testing validation loss
train_loss = map(lambda x: x['main/loss'], data)
validation_loss = map(lambda x: x['validation/main/loss'], data)

# plot training and validation error
plt.figure()
plt.plot(np.arange(len(train_loss)),np.transpose(np.vstack([train_loss,validation_loss])))
plt.legend(['training', 'validation'])
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.show()