# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 14:28:34 2016

@author: hagay
"""
from sklearn.svm import SVC , SVR
from sklearn.metrics.pairwise import chi2_kernel
from scipy.spatial.distance import euclidean



X = []
Y = []
for d in ds:
    X.append(d["vec"])
    Y.append(d["label"])
    
X = np.asarray(X)
Y = np.asarray(Y)


svm = SVR(kernel=chi2_kernel).fit(X, Y)

len_meas = 200
avg_dis = 0.0
for i in range(len_meas):
    print  svm.predict(X[i])[0] ,  Y[i]
    avg_dis += np.abs(svm.predict(X[i])[0] -  Y[i])
    
print avg_dis/len_meas
