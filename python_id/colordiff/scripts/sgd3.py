#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 23:38:13 2017

@author: yakir
"""
import random
import  numpy as np

def L2(pair):
    v1 = pair[:3]
    v2 = pair[3:]
    return np.sqrt(np.sum([(el1-el2)**2 for el1,el2 in zip(v1,v2)]))

def L1(pair):
    v1 = pair[:3]
    v2 = pair[3:]
    return np.abs(np.sum([(el1-el2)**2 for el1,el2 in zip(v1,v2)]))    
# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return yhat

def gen_ds(d_num = 1000, n_feat = 100):
        dataset = []
        for d in range(d_num):
            v = [random.gauss(0,0.2) for _ in range(n_feat)]
            label = random.randint(0,3)
            v.append(label)
            dataset.append(v)
        return dataset
    
# Estimate linear regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0])-6)]
    for epoch in range(n_epoch):
        if epoch > 1000:
            l_rate = l_rate*0.1
        sum_error = 0
        for row in train:
            pair = row[:6]
            del row[:6]
            yhat = predict(row, coef)
            error = yhat - row[-1] + L2(pair)
            sum_error += error**2
            coef[0] = coef[0] - l_rate * error
            for i in range(len(row)-1):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        print 
    return coef

# Calculate coefficients
dataset = gen_ds()
l_rate = 0.001
n_epoch = 5
w = coefficients_sgd(dataset, l_rate, n_epoch)
print(w)

    
    