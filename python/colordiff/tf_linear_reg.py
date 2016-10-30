# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 15:17:34 2016

@author: yakir
"""

#!/usr/bin/env python

from keras import backend as K
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from keras import objectives
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

tf = K.tensorflow_backend.tf
        
def find_non_zero(W):
    for i in range(len(W)):
        if W[i] >0.000000001:
            print i , W[i]
            
            
def model(X, w):
    return tf.mul(X, w) # lr is just X*w so this model line is pretty simple

trX = np.asarray(X_tr)
trY = np.asarray(Y_tr)
#trX = trX.transpose()
X = tf.placeholder("float") # create symbolic variables
Y = tf.placeholder("float")

w = tf.Variable(np.zeros(len(trX[0,:]) , dtype = np.float32), name="weights") # create a shared variable (like theano.shared) for the weight matrix
y_model = model(X, w)

beta = 16000
learning_rate = 0.005
n_samples = len(X_tr)

# Loss function using L2 Regularization
regularizer = tf.nn.l2_loss(w)


# Mean squared error
cost = tf.reduce_sum((beta*tf.pow(y_model-Y, 2))/(2*n_samples)) + regularizer
# Gradient descent
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize variables (in this case just variable W)
    tf.initialize_all_variables().run()
    
    for i in range(20):
        print "training iter num:",i
        for (x, y) in zip(trX, trY):
            find_non_zero(x)
            print y
            sess.run(train_op, feed_dict={X: x, Y: y})
            print 
            
        W = np.asarray(w.eval())
        find_non_zero(W)
        
        
        
avg_dis = 0.0
for x in X_te:
    ind = X_te.index(x)
    find_non_zero(x)
    print ind , np.dot(x,W) , Y_te[ind]
    avg_dis += np.abs(np.dot(x,W) - Y_te[ind])
    
    print 
    
    
avg_dis = avg_dis/len(X_te)

print "avg dis = ", avg_dis
