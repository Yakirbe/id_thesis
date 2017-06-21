#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 23:38:13 2017

@author: yakir
"""
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
import  numpy as np
from prepare_ds import de_pair

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
    yhat = coefficients[0] * row[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i + 1]
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
def train(tr_data, labels, l_rate, n_epoch, weight = "", reg = ""):
    if weight:
        coef = weight
    else:
        coef = [0.0 for i in range(len(tr_data[0])-6)]
    for epoch in range(n_epoch):
#        if epoch%500 == 0:
#            l_rate = l_rate*0.5
        sum_error = 0
        index = 0
        for row in tr_data:
            l = labels[index]
            pair = row[:6]
            #print pair
            emb_vec = row[6:]
            yhat = np.dot(emb_vec, coef)
            if not reg:
<<<<<<< HEAD
                rg = L2(pair) 
            error = yhat - l + 0.01*rg
            sum_error += error**2
            for i in range(len(emb_vec)):
                #coef[i + 1] = np.max(0.0,coef[i + 1] - l_rate * error * emb_vec[i])
=======
                #rg = L1(pair) 
                rg = de_pair(pair)
            error = yhat - l + 0.01*rg
            #print "error = ", error, "rg = " , rg, "pred = ", yhat, "true = ", l
            sum_error += error**2
            for i in range(len(emb_vec)):
>>>>>>> 9e89fafcff4033420920eb4eaba6d32ce40184a5
                coef[i] =coef[i] - l_rate * error * emb_vec[i]
            index += 1
        print('epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        #print
    return coef
    
    
def test(test, labels, w, method = "L2"):
    
    i = 0
    yp_l = []
    yt_l = []
    for t in test:
        pair = t[:6]
        sample = t[6:]
<<<<<<< HEAD
        rg = L2(pair) 
=======
        rg = de_pair(pair)
>>>>>>> 9e89fafcff4033420920eb4eaba6d32ce40184a5
        y_pred = np.dot(sample, w) + 0.01*rg
        y_true = labels[i]
        yt_l.append(y_true)
        yp_l.append(y_pred)
        print "pred = ", y_pred , "true = ", y_true, "l2 = " , L2(pair)
        i += 1
    print "mse =",  mean_squared_error(yt_l, yp_l)
    print "mae =",  mean_absolute_error(yt_l, yp_l)
    errors = [mean_absolute_error(yt_l, yp_l) ,mean_squared_error(yt_l, yp_l)]
    return errors


if __name__ == "__main__":
    # Calculate coefficients
    l_rate = 0.001
    n_epoch = 300
    #w = train(X_tr,Y_tr, l_rate, n_epoch)
    #l2_error = test(X_te, Y_te, w)
    #l2_error = test(X_tr, Y_tr, w)
    
    print predict([1,2,3],[1,2,4])
    print np.dot([1,2,3],[1,2,4])