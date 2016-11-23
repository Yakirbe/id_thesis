#import tensorflow as tf


from keras import backend as K
import theano.tensor as T
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from keras import objectives
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Generating the sample data from make_blobs
# This particular setting has one distinct cluster and 3 clusters placed close
# together.
X, y = make_blobs(n_samples=500,
                  n_features=1,
                  centers=4,
                  cluster_std=1,
                  center_box=(-10.0, 10.0),
                  shuffle=True,
                  random_state=1)  # For reproducibility


def get_opt_k(X ,max_range):
    range_n_clusters = range(2,max_range)
    max_sil = -1
    max_n = 0
    for n_clusters in range_n_clusters:
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed clusters
        
        silhouette_avg = silhouette_score(X, cluster_labels)
       
        if max_sil<silhouette_avg:
            
           max_sil = silhouette_avg 
           max_n = n_clusters
           
    return max_n
    


def custom_(y_true, y_pred):
    C = 1
    
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.mean(y_true * y_pred, axis=-1)

    
sess = K.tensorflow_backend.tf.Session()    
K.set_session(sess)

y_true = K.variable(value=np.random.rand(10))
y_pred = K.variable(value=np.random.rand(10))
ans1 = objectives.hinge(y_true,y_pred)
ans2 = objectives.mean_absolute_error(y_true,y_pred)
ans3 = objectives.mean_squared_error(y_true,y_pred)
ans4 = objectives.poisson(y_true,y_pred)
ans5 = custom_(y_true,y_pred)
with sess.as_default():
    print type(y_true)
    print type(y_pred)
    print y_pred.eval()
    print y_true.eval()
    print ans1.eval()
    print ans2.eval()
    print ans3.eval()
    print ans4.eval()
    print ans5.eval()
