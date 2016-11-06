# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 17:50:20 2016

@author: yakir
"""
import os , sys , random , copy ,  time
from sklearn import svm , cluster , multiclass
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from operator import add
from scipy.sparse import csr_matrix
plt.close("all")
import json



def norm_0_1(val , m_min , m_max):
        """
        this function normalizes between 0 - 1
        """
        if val == m_min:
            return 0.0
        elif val == m_max:
            return 1.0
        else:
            return (val - m_min)/(m_max - m_min)
            

def norm_vec( v_vec , cube):
        n_vec = []
        for v in v_vec:
            n_vec.append(norm_0_1(v , cube[v_vec.index(v)][0] , cube[v_vec.index(v)][1]))  
        return n_vec
        
        
        
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
    
    
    
def using_indexed_assignment(x):
    result = np.empty(len(x), dtype=int)
    temp = np.argsort(x)
    result[temp] = np.arange(len(x))
    result = [np.max(result) - r for r in result]
    return list(result)
        
        
def find_simplex( v_vec):
        args = using_indexed_assignment(v_vec)
        refs = []
        for r in range(len(v_vec) + 1):
            ref = [0]*len(v_vec)
            
            for ar in args:
                if r > ar:
                    ref[args.index(ar)] = 1
            refs.append(ref)
        return refs



def interpolate(tr_ds_clf , cvecs):
    
    ds_json = []
    for v_vec in tr_ds_clf:
        ds_sample = {}
    
        ds_sample["v_vec"] = v_vec["vec"]
        ds_sample["label"] = v_vec["label"]
        #print v_vec , label , label_num   
        
        lows = []
        
        v_ind = 0
        for v_i in v_vec["vec"]:
            c_vec = cvecs[v_ind]
            low_bound = np.argmin([np.abs(x - v_i) for x in c_vec])
            
            if v_i < c_vec[low_bound] and low_bound <> 0:
                low_bound -= 1
                
            lows.append(low_bound)
            
            v_ind += 1 
        ds_sample["lows"] = lows
        ##find hypercube
        hypercube = []
        
        l_ind = 0
        for l in lows:
            c_vec = cvecs[l_ind]
            hypercube.append([c_vec[l] ,c_vec[l + 1]])
            
            l_ind += 1
        ds_sample["hypercube"] = hypercube
        #dataset_labeled[samp]["cube"] = cube
                                        
        ##normalize
        n_vec = norm_vec(v_vec["vec"] , hypercube)
        ds_sample["n_vec"] = n_vec
    
        ##find simplex
        ref = find_simplex(n_vec)
        ##calc coefficients    
        coef_vec = list(np.linalg.solve(np.transpose(np.asarray(ref[1:])) , n_vec))
        coef_vec.insert( 0 , 1 - sum(coef_vec) )
        
        assert all(i >= 0.0 for i in coef_vec)
        assert np.abs(np.sum(coef_vec) - 1.0) < 0.0000000001 , "sum is not 1, but:{}".format(str(np.sum(coef_vec)))
        ds_sample["refs"] = ref
        ds_sample["coef_vec"] = coef_vec
        #print  ds_sample["coef_vec"] , ds_sample["label"]
        
        ds_json.append(ds_sample)
        
    print "dataset interpolated\n"
    
    
#    for v in ds_json:
#        print v["coef_vec"]
    
    return ds_json
    
    

def embed(dataset ,cvecs):
    
    print "embed..."
    ds_out = dataset
    
    for samp in range(len(dataset)):
        
#        if samp%100 == 0:
#            print dataset[samp]["coef_vec"]
#            print dataset[samp]["v_vec"]
#            print 
        
        C = max([len(cv) for cv in cvecs])
        emb_vec = [0.0]*C**(len(dataset[samp]["v_vec"]))
        
        refs_bounded = []
        
        for r in dataset[samp]["refs"]:
            refs_bounded.append(map(add, r, dataset[samp]["lows"]))
        
        for j in range(len(refs_bounded)):
            ind = 0
            for i in range(len(refs_bounded[0])-1): 
                ind += C*refs_bounded[j][i]
                #print j,i,ind
                
            ind += refs_bounded[j][-1]
            #print ind
            #print
            
            emb_vec[ind] = dataset[samp]["coef_vec"][j]
        emb_vec = csr_matrix(emb_vec , dtype=np.float64)
        ds_out[samp]["embedded"] = emb_vec
            
    print "dataset embedded\n"
    
    return ds_out    





####################################################################################################################################        



# labels indices

data_fn = "/home/yakir/idd/tex_thesis/id_thesis/python/colordiff/rgb_set.json"
# load dataset ----------------------------------------------------------------
        
with open(data_fn) as f:
    ds = f.readlines()

#concat vecs
ds_new = []    
for d in ds:
    ds_new.append(json.loads(d.replace("'",'"')))
ds_new = ds_new[0]    
ds= []
for d in ds_new:
    vec_app = d["v1"] + d["v2"]
    vec_app = [int(255*el) for el in vec_app]
    ds.append({"vec":vec_app , "label":d["dis"]})
    

#shuffle
ds = random.sample(ds , len(ds))

# set discrete extreme vals ---------------------------------------------------

dis_min = 1000
dis_max = -1

for i in range(len(ds)):
    if dis_max < np.max(ds[i]["vec"]):
        dis_max = np.max(ds[i]["vec"])
        
    if dis_min > np.min(ds[i]["vec"]):
        dis_min = np.min(ds[i]["vec"])
        
print "dismin = {} , dismax = {}".format(dis_min,dis_max)
        
# classification --------------------------------------------------------------

# separate to train-test
    
tr_set = []
te_set = []

# generate dataset (train and test sets)

rate_tr = 0.8
tr_set = ds[:int(rate_tr*len(ds))]
te_set = ds[int(rate_tr*len(ds)):]


# generate dis vecs
   
kmeans_flag = True     
min_max_margin = 0.01       
  
if kmeans_flag:
    cvecs = []
    for dim in range(len(ds[0]["vec"])):
        X = []
        for t in tr_set:
            X.append([t["vec"][dim]])
            
        ds_temp =(np.array(X).reshape(-1, 1))
        X = np.asarray(X)
        C = get_opt_k(X ,6)
        k_means = cluster.KMeans(C)
        k_means.fit(np.asarray(ds_temp))
        c_vec = (k_means.cluster_centers_)
        
        c_vec = sorted([cv[0] for cv in c_vec])
        c_vec.insert(0 , dis_min - min_max_margin)
        c_vec.append(dis_max + min_max_margin)
        
        cvecs.append(c_vec)
        
print cvecs        
print "c vector genarated\n"   


# interpolation ---------------------------------------------------------------

print "interpolate.........."

train_ds_json = interpolate(tr_set , cvecs)
test_ds_json = interpolate(te_set , cvecs)

# embed -----------------------------------------------------------------------

train_ds_json = embed(train_ds_json , cvecs)
test_ds_json = embed(test_ds_json , cvecs)

#concat groups

#train set

tr_emb = []
tr_lab = []
for i in range(len(train_ds_json)):
    tr_emb.append(list(train_ds_json[i]["embedded"].toarray()[0]))
    tr_lab.append(train_ds_json[i]["label"])
        
#test set
        
te_emb = []
te_lab = []
for i in range(len(test_ds_json)):
    te_emb.append(list(test_ds_json[i]["embedded"].toarray()[0]))
    te_lab.append(test_ds_json[i]["label"])

    
# train -----------------------------------------------------------------------
    
def remove_zero_col(X_tr):
    
    X_tr = np.asarray(X_tr)
    X_temp = X_tr.transpose()
    X_temp = X_temp[~(X_temp==0).all(1)]
    X_temp = X_temp.transpose()
    X_tr = X_temp.tolist()
    
    return X_tr
    
    
    
X_tr = [s for s in tr_emb]
Y_tr = [s for s in tr_lab]

X_tr = remove_zero_col(X_tr)


def zero_pad_lower(X_tr ,X_te):
    
    l1 = len(X_te[0]) 
    l2 = len(X_tr[0])
    
    print l1, l2
    if l1  ==  l2:
        print "equal"
        return X_tr ,X_te
    elif l1 > l2:
        xout = []
        for x in X_tr:
            xout.append(x + [0.0]*int(l1-l2))
        print "l1>l2"
        return xout ,X_te    
    elif l2 > l1:
        xout = []
        for x in X_te:
            xout.append(x + [0.0]*int(l2-l1))        
        print "l2>l1"
        return X_tr ,xout        
    


#
#print "training..."
#clf = svm.SVR(C = 1e5 ,  kernel='linear' , cache_size  = 200)
#clf.fit(X_tr, Y_tr) 
#
#print "training done\n"
#
#
## test ------------------------------------------------------------------------
#
#        
#print "testing..."
#
X_te = [s for s in te_emb]
Y_te = [s for s in te_lab]


X_te = remove_zero_col(X_te)

X_tr ,X_te = zero_pad_lower(X_tr ,X_te)

print len(X_te[0]) , len(X_tr[0])
#tst_set = len(X_te)
#
#avg_dis = 0.0
#i = 0
#for x in X_te:
#    print
#    
#    dis = np.abs(Y_te[i] - clf.predict([x])[0])
#    avg_dis += dis
#    
#    if i%2 == 0:
#        print  i , "done from" , len(X_te)
#        print  Y_te[i] , clf.predict([x])[0]
#        print "distance = ", dis
#    i+=1
#print "avg dis = ", avg_dis/len(X_te)        

