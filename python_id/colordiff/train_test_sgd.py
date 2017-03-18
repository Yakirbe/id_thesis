#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 23:38:13 2017

@author: yakir
"""
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
import  numpy as np


def L2(pair):
    v1 = pair[:3]
    v2 = pair[3:]
    return np.sqrt(np.sum([(el1-el2)**2 for el1,el2 in zip(v1,v2)]))


def L1(pair):
    v1 = pair[:3]
    v2 = pair[3:]
    return np.sqrt(np.sum([np.abs(el1-el2) for el1,el2 in zip(v1,v2)]))   
    
    
# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return yhat

def gen_ds(d_num = 5000, n_feat = 100):
        dataset = []
        for d in range(d_num):
            v = [random.gauss(0,0.2) for _ in range(n_feat)]
            label = random.randint(0,3)
            v.append(label)
            dataset.append(v)
        return dataset
    
# Estimate linear regression coefficients using stochastic gradient descent
def train(train, labels, l_rate, n_epoch, weight = ""):
    if weight:
        coef = weight
    else:
        coef = [0.0 for i in range(len(train[0])-6)]
    for epoch in range(n_epoch):
#        if epoch > 500:
#            l_rate = l_rate*0.1
        sum_error = 0
        index = 0
        for row in train:
            l = labels[index]
            pair = row[:6]
            emb_vec = row[6:]
            yhat = predict(emb_vec, coef)
            error = yhat - l + 0.001*L2(pair)
            sum_error += error**2
            coef[0] = coef[0] - l_rate * error *emb_vec[0]
            for i in range(len(emb_vec)-1):
                #coef[i + 1] = np.max(0.0,coef[i + 1] - l_rate * error * emb_vec[i])
                coef[i + 1] =coef[i + 1] - l_rate * error * emb_vec[i]
            index += 1
        print('epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        print
    return coef
    
    
def test(test, labels, w, method = "L2"):
    
    avg = 0.0
    i = 0
    yp_l = []
    yt_l = []
    for t in test:
        pair = t[:6]
        sample = t[6:]
        y_pred = np.dot(sample, w)
        y_true = labels[i]
        yt_l.append(y_true)
        yp_l.append(y_pred)
        #print "pred = ", y_pred , "true = ", y_true
        if method == "L2":
            error = (y_pred - y_true + 0.001*L2(pair))**2 
            avg += error
        elif method == "L1":
            error = np.abs(y_pred - y_true + 0.001*L2(pair))
            avg += error
        i += 1
    print "hand made =",  avg/len(test)
    print "mse =",  mean_squared_error(yt_l, yp_l)
    print "mae =",  mean_absolute_error(yt_l, yp_l)


if __name__ == "__main__":
    # Calculate coefficients
    l_rate = 0.001
    n_epoch = 300
    #w = train(X_tr,Y_tr, l_rate, n_epoch)
    l2_error = test(X_te, Y_te, w)
    #l2_error = test(X_tr, Y_tr, w)
    print l2_error
