# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 17:50:20 2016

@author: yakir
"""
import os , sys , random , copy ,  time
from sklearn import svm , cluster , multiclass
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



def interpolate(tr_ds_clf , tr_labels_clf):
    
    ds_json = []
    for v_vec in tr_ds_clf:
        ds_sample = {}
    
        ds_sample["v_vec"] = v_vec
    
        label = tr_labels_clf[tr_ds_clf.index(v_vec)]
        label_num = label
        
        ds_sample["label"] = label_num
        #print v_vec , label , label_num   
        
        lows = []
        for v_i in v_vec:
            low_bound = np.argmin([np.abs(x - v_i) for x in c_vec])
            
            if v_i < c_vec[low_bound] and low_bound <> 0:
                low_bound -= 1
                
            lows.append(low_bound)
            
        ds_sample["lows"] = lows
    
        ##find hypercube
        hypercube = []
        for l in lows:
            hypercube.append([c_vec[l] ,c_vec[l + 1]])
                
        ds_sample["hypercube"] = hypercube
        #dataset_labeled[samp]["cube"] = cube
                                        
        ##normalize
        n_vec = norm_vec(v_vec , hypercube)
        ds_sample["n_vec"] = n_vec
    
        ##find simplex
        ref = find_simplex(n_vec)
        ##calc coefficients    
        coef_vec = list(np.linalg.solve(np.transpose(np.asarray(ref[1:])) , n_vec))
        coef_vec.insert( 0 , 1 - sum(coef_vec) )
        
        ds_sample["refs"] = ref
        ds_sample["coef_vec"] = coef_vec
        #print  ds_sample["coef_vec"] , ds_sample["label"]
        
        ds_json.append(ds_sample)
        
    print "dataset interpolated\n"
    
    
#    for v in ds_json:
#        print v["coef_vec"]
    
    return ds_json
    

def embed(dataset , C = 4):
    
    print "embed..."
    ds_out = dataset
    
    for samp in range(len(dataset)):
        #print samp
    
        emb_vec = [0.0]*len(c_vec)**(len(dataset[samp]["v_vec"]))
        
        refs_bounded = []
        
        for r in dataset[samp]["refs"]:
            refs_bounded.append(map(add, r, dataset[samp]["lows"]))
        
        #print dataset[samp]["lows"]
        #print refs_bounded
        for j in range(len(refs_bounded)):
            ind = 0
            for i in range(len(refs_bounded[0])-1): 
                ind += C*refs_bounded[j][i]
            ind += refs_bounded[j][-1]
            
            emb_vec[ind] = dataset[samp]["coef_vec"][j]
        #print dataset[samp]["coef_vec"]    
        emb_vec = csr_matrix(emb_vec , dtype=np.float64)
        ds_out[samp]["embedded"] = emb_vec
        #print emb_vec , "\n"
            
    print "dataset embedded\n"
    
    return ds_out    

        

# labels indices

path_ims = "/home/yakir/idd/datasets/retargeting/"
labels = sorted(os.listdir(path_ims))


# load dataset ----------------------------------------------------------------
        
with open("fn.json") as f:
    ds = f.readlines()

ds_new = []    
for d in ds:
    ds_new.append(json.loads(d.replace("'",'"')))

# set groups for pair matching tasks -----------------------
    
        
grps_pool = [(55,56) , (57,58) , (35,74) , (64,65) , (60,61) , 
        (30,31) , (62,63) , (44,45) , (45,67) , (75,77) , (45,80)]


# rearrange dataset in pairs format by groups -----------------------
        
# match label = -1 , diffenret label = 1

n_match = 2500
ds_match = []
ds_diff = []
match_stop = False
diff_stop = False

while not match_stop or not diff_stop:
    
    ind1 = random.choice(range(len(ds_new)))
    ind2 = random.choice(range(len(ds_new)))
    l1 = ds_new[ind1]["label"]
    l2 = ds_new[ind2]["label"]
    if l1 == l2:
        if not match_stop:
            ds_match.append([ds_new[ind1]["vec"] , ds_new[ind2]["vec"]])
    else:
        if not diff_stop:
            ds_diff.append([ds_new[ind1]["vec"] , ds_new[ind2]["vec"]])
        
    if len(ds_match) >= n_match:
        match_stop = True
        
    if len(ds_diff) >= n_match:
        diff_stop = True

# extract group pair data ----------------------------------------------------------    

pairs_num = 8

tr_grp = []
te_grp = []

for i in range(pairs_num):
    
    pairs_inds = random.sample(grps_pool , 2)
    grp_ds = []
    for samp in ds_match:
        vec = [samp[0][pairs_inds[0][0]] , samp[0][pairs_inds[0][1]] , samp[1][pairs_inds[1][0]] , samp[1][pairs_inds[1][1]]]
        label = -1
        grp_ds.append({"vec":vec , "label":label})
    
    for samp in ds_diff:
        vec = [samp[0][pairs_inds[0][0]] , samp[0][pairs_inds[0][1]] , samp[1][pairs_inds[1][0]] , samp[1][pairs_inds[1][1]]]
        label = 1
        grp_ds.append({"vec":vec , "label":label})

    ds_new = grp_ds
#    grp_dss.append(ds_new)


    # set discrete extreme vals ---------------------------------------------------
    
    dis_min = 1000
    dis_max = -1
    
    for i in range(len(ds_new)):
        if dis_max < np.max(ds_new[i]["vec"]):
            dis_max = np.max(ds_new[i]["vec"])
            
        if dis_min > np.min(ds_new[i]["vec"]):
            dis_min = np.min(ds_new[i]["vec"])
            
    # classification --------------------------------------------------------------
    
    # separate to train-test
        
    test_pr = 0.2
    tr_set = []
    te_set = []
    for d in ds_new:
        
        if ds_new.index(d) <= test_pr*len(ds_new):
            
        else:
        if random.uniform(0,1) <= test_pr:
            te_set.append(d)
        else:
            tr_set.append(d)
            
    # shuffle
    tr_set = random.sample(tr_set , len(tr_set))
    te_set = random.sample(te_set , len(te_set))
    
    # generate dataset (train and test sets)


    tr_ds_clf = []
    tr_labels_clf = []
    
    te_ds_clf = []
    te_labels_clf = []

    for i in range(len(tr_set)):
        samp_vec = tr_set[i]["vec"]
        tr_labels_clf.append(str(tr_set[i]["label"]))
        tr_ds_clf.append(samp_vec)
        
        
    for i in range(len(te_set)):
        samp_vec = te_set[i]["vec"]
        te_labels_clf.append(str(te_set[i]["label"]))
        te_ds_clf.append(samp_vec)


    # generate dis vecs
        
    C = 2 
    kmeans_flag = True     
    min_max_margin = 0.01         
    if kmeans_flag:
        ds_temp =(np.array(tr_ds_clf).reshape(-1, 1))
        k_means = cluster.KMeans(C)
        k_means.fit(np.asarray(ds_temp))
        c_vec = (k_means.cluster_centers_)
        
        c_vec = sorted([cv[0] for cv in c_vec])
        c_vec.insert(0 , dis_min -min_max_margin)
        c_vec.append(dis_max + min_max_margin)
    print "c vector genarated\n"   


# interpolation ---------------------------------------------------------------

    print "interpolate.........."
    
    train_ds_json = interpolate(tr_ds_clf , tr_labels_clf)
    test_ds_json = interpolate(te_ds_clf , te_labels_clf)

# embed -----------------------------------------------------------------------

    train_ds_json = embed(train_ds_json , C)
    test_ds_json = embed(test_ds_json , C)
    
    tr_grp.append(train_ds_json)
    te_grp.append(test_ds_json)


#concat groups



#train set

print "preparing training set"

tr_grp_emb = [[]]*np.max([len(i) for i in tr_grp])
tr_grp_label = [0]*np.max([len(i) for i in tr_grp])

for g in tr_grp:
    for i in range(len(tr_grp_emb)):
        tr_grp_emb[i] = tr_grp_emb[i] + list(g[i]["embedded"].toarray()[0])
        tr_grp_label[i] = g[i]["label"]
        
#test set

te_grp_emb = [[]]*np.max([len(i) for i in te_grp])
te_grp_label = [0]*np.max([len(i) for i in te_grp])
        
for g in te_grp:
    for i in range(len(te_grp_emb)):
        te_grp_emb[i] = te_grp_emb[i] + list(g[i]["embedded"].toarray()[0])
        te_grp_label[i] = g[i]["label"]
    
    
# train -----------------------------------------------------------------------
    
print "training..."
X_tr = [s for s in tr_grp_emb]
Y_tr = [s for s in tr_grp_label]

clf = svm.SVC(C = 1e5 , decision_function_shape='ovr' , kernel='rbf' , cache_size  = 200)
clf.fit(X_tr, Y_tr) 

print "training done\n"


# test ------------------------------------------------------------------------


        
print "testing..."

X_te = [te_grp_emb[samp] for samp in range(len(te_grp_emb))]
Y_te = [te_grp_label[samp] for samp in range(len(te_grp_label))]

tst_set = len(X_te)
tp = 0.0
fp = 0.0
for x in X_te:
    
    if X_te.index(x)%10 == 0:
        print  X_te.index(x) , "done from" , len(X_te)
        print  Y_te[X_te.index(x)] , clf.predict([x])[0]
        
    if Y_te[X_te.index(x)] == clf.predict([x])[0]:
        tp += 1
    else:
        fp += 1
print "true positive = " , tp , "of" , tst_set
print "false positive = " , fp , "of" , tst_set
print "tp % = " , 100*tp/tst_set
print "fp % = " , 100*fp/tst_set
print "testing done\n\n\n\n"
