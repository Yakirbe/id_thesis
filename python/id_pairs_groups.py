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
    

def embed(dataset ,c_vec ):
    
    C = len(c_vec) - 2
    
    print "embed..."
    ds_out = dataset
    
    for samp in range(len(dataset)):
    
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


num_per_cls = 2500
C = 4
###############################################################################

# labels indices

path_ims = "/home/yakir/idd/datasets/retargeting/"
path_ims = "C:\ofir_stuff_idd\datasets\dataset-20110512"
labels = sorted(os.listdir(path_ims))


class ID:
    
    def __init__(ds = [] , grps = [] , pairs = True , C = 4 , d_min = 1000 ,
                 d_max = -1, test_pr = 0.2 , min_max_margin = 0.01 , 
                 c_vec = [] , tr_set = [] , te_set = [] , 
                 labeled_size = 2500):
        
        self.ds = ds
        self.grps = grps
        self.pairs = pairs
        self.C = C
        self.d_min = d_min
        self.d_max = d_max
        self.test_pr = test_pr
        self.min_max_margin = min_max_margin
        self.labeled_size = labeled_size
        self.c_vec = c_vec
        self.tr_set = tr_set
        self.te_set = te_set
    
    
    
# load dataset ----------------------------------------------------------------
        
with open("fn.json") as f:
    ds = f.readlines()

ds_new = []    
for d in ds:
    ds_new.append(json.loads(d.replace("'",'"')))

ds_new = 5*ds_new

# set groups for classification and pair matching tasks -----------------------
    
grps_clf1 = [(95,96) , (29,26) , (88,87), (63,62) , (50,49)  , (42,43) , (60,61) , (90,92) , (40,41)]
grps_clf2 = [(107,108) , (27,28) , (89,86) , (64,65) , (70,69)  , (44,45) , (50,51) , (93,91) , (40,41)]
grps_clf = [ grps_clf1 , grps_clf2 ]

# set discrete extreme vals ---------------------------------------------------

dis_min = 1000
dis_max = -1

for i in range(len(ds_new)):
    if dis_max < np.max(ds_new[i]["vec"]):
        dis_max = np.max(ds_new[i]["vec"])
        
    if dis_min > np.min(ds_new[i]["vec"]):
        dis_min = np.min(ds_new[i]["vec"])
        
# classification --------------------------------------------------------------

#get pairs of sims-diffs

sims = []
difs = []

stop_sim = False
stop_dif = False

while (not stop_dif) or (not stop_sim):
    
    v1 = random.choice(ds_new)
    v2 = random.choice(ds_new)
    
    if v1 == v2:
        continue
    else:
        if v1["label"] == v2["label"]: 
            if not stop_sim:
                sims.append((v1["vec"] , v2["vec"] , -1))
                if len(sims) >= num_per_cls:
                    stop_sim = True
        else:
            if not stop_dif:
                difs.append((v1["vec"] , v2["vec"] , 1))
                if len(difs) >= num_per_cls:
                    stop_dif = True

ds_out = sims+difs
ds_out = random.sample(ds_out , len(ds_out))

tr_grp = []
te_grp = []

pairs_len = 6

grps_clf[0] = grps_clf[0][:pairs_len]
grps_clf[1] = grps_clf[1][:pairs_len]
for g1 in grps_clf[0]:
    g2 = grps_clf[1][grps_clf[0].index(g1)]
    ds_out_pairs = []
    
    #get group indices data
    ds_out_pairs =  []
    for d in ds_out:
        sample = [d[0][g1[0]] , d[0][g1[1]] , d[1][g2[0]] , d[1][g2[1]]]
        ds_out_pairs.append({"vec":sample , "label":d[2]})
        
    # separate to train-test
    
    test_pr = 0.2
    
    te_set = ds_out_pairs[:int(test_pr*len(ds_out_pairs))]
    tr_set = ds_out_pairs[int(test_pr*len(ds_out_pairs)):]

    # generate dataset (train and test sets)

    tr_ds_clf = []
    tr_labels_clf = []

    te_ds_clf = []
    te_labels_clf = []
    
    for i in range(len(tr_set)):
        v = tr_set[i]["vec"]
        tr_labels_clf.append(str(tr_set[i]["label"]))
        tr_ds_clf.append(v)
        
    for i in range(len(te_set)):
        v = te_set[i]["vec"]
        te_labels_clf.append(str(te_set[i]["label"]))
        te_ds_clf.append(v)

    # generate dis vecs
        
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
    
    train_ds_json = embed(train_ds_json ,c_vec)
    test_ds_json = embed(test_ds_json ,c_vec)
    
    tr_grp.append(train_ds_json)
    te_grp.append(test_ds_json)

#concat groups

# train set
tr_grp_emb = [[]]*np.min([len(vr) for vr in tr_grp])
tr_grp_label = [0]*np.min([len(ve) for ve in tr_grp])

for g in tr_grp:
    for i in range(len(tr_grp_emb)):
        tr_grp_emb[i] = tr_grp_emb[i] + list(g[i]["embedded"].toarray()[0])
        tr_grp_label[i] = g[i]["label"]
        
# test set

te_grp_emb = [[]]*np.min([len(v) for v in te_grp])
te_grp_label = [0]*np.min([len(v) for v in te_grp])
        
for g in te_grp:
    for i in range(len(te_grp_emb)):
        te_grp_emb[i] = te_grp_emb[i] + list(g[i]["embedded"].toarray()[0])
        te_grp_label[i] = g[i]["label"]
    
    
# train -----------------------------------------------------------------------
    
print "training..."
X_tr = [s for s in tr_grp_emb]
Y_tr = [s for s in tr_grp_label]

clf = svm.SVC(C = 1e5 , decision_function_shape='ovo' , kernel='rbf' , cache_size  = 200)
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

