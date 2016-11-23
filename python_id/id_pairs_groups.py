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


class ID:
    
    def __init__(self, ds = [] , grps = [] , pairs = True , C = 4 , d_min = 1000 ,
                 d_max = -1, test_pr = 0.2 , min_max_margin = 0.01 , 
                 c_vec = [] , tr_set = [] , te_set = [] , grps_len = 0 ,
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
        self.grps_len = grps_len
        
        
    def set_ext_vals(self,dis_min = 1000 , dis_max = -1):
        
        dmin = dis_min
        dmax = dis_max
        
        for i in range(len(self.ds)):
            if dmax < np.max(self.ds[i]["vec"]):
                dmax = np.max(self.ds[i]["vec"])
                
            if dmin > np.min(self.ds[i]["vec"]):
                dmin = np.min(self.ds[i]["vec"])
                
        self.d_min = dmin
        self.d_max = dmax
        
        
    def set_pairs_ds(self):
        
        sims = []
        difs = []

        stop_sim = False
        stop_dif = False
        
        while (not stop_dif) or (not stop_sim):
    
            v1 = random.choice(self.ds)
            v2 = random.choice(self.ds)
            
            if v1 == v2:
                continue
            else:
                if v1["label"] == v2["label"]: 
                    if not stop_sim:
                        sims.append((v1["vec"] , v2["vec"] , 0))
                        if len(sims) >= self.labeled_size:
                            stop_sim = True
                else:
                    if not stop_dif:
                        difs.append((v1["vec"] , v2["vec"] , 1))
                        if len(difs) >= self.labeled_size:
                            stop_dif = True

        ds_out = sims+difs
        ds_out = random.sample(ds_out , len(ds_out))
        
        self.ds = ds_out
        
        
    def main_sngl_run(self):
        
        tr_set = []
        te_set = []
        for d in self.ds:
            if random.uniform(0,1) <= self.test_pr:
                te_set.append(d)
            else:
                tr_set.append(d)
                
        # shuffle
        tr_set = random.sample(tr_set , len(tr_set))
        te_set = random.sample(te_set , len(te_set))
        
        tr_grp = []
        te_grp = []
                
        for grp_ind in range(len(self.grps)):
    
            tr_ds_clf = []
            tr_labels_clf = []
        
            te_ds_clf = []
            te_labels_clf = []
            
            for i in range(len(tr_set)):
                samp_vec = tr_set[i]["vec"]
                tr_labels_clf.append(str(tr_set[i]["label"]))
                v = [samp_vec[s] for s in self.grps[grp_ind]]
                tr_ds_clf.append(v)
                
                
            for i in range(len(te_set)):
                samp_vec = te_set[i]["vec"]
                te_labels_clf.append(str(te_set[i]["label"]))
                v = [samp_vec[s] for s in self.grps[grp_ind]]
                te_ds_clf.append(v)
        
        
            # generate dis vecs
                
            kmeans_flag = True     
                    
            if kmeans_flag:
                ds_temp =(np.array(tr_ds_clf).reshape(-1, 1))
                k_means = cluster.KMeans(self.C)
                k_means.fit(np.asarray(ds_temp))
                c_vec = (k_means.cluster_centers_)
                
                c_vec = sorted([cv[0] for cv in c_vec])
                c_vec.insert(0 , self.d_min - self.min_max_margin)
                c_vec.append(self.d_max + self.min_max_margin)
                
            self.c_vec = c_vec
            print "c vector genarated\n"      
        
        
            # interpolation ---------------------------------------------------------------
            
            print "interpolate.........."
            
            train_ds_json = self.interpolate(tr_ds_clf , tr_labels_clf)
            test_ds_json = self.interpolate(te_ds_clf , te_labels_clf)
            
            # embed -----------------------------------------------------------------------
            
            train_ds_json = embed(train_ds_json ,self.c_vec)
            test_ds_json = embed(test_ds_json ,self.c_vec)
            
            tr_grp.append(train_ds_json)
            te_grp.append(test_ds_json)
            
        #concat groups

        # train set
        tr_grp_emb = [[]]*np.min([len(vr) for vr in tr_grp])
        tr_grp_label = [0.5]*np.min([len(ve) for ve in tr_grp])
        
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
        
        return 100*fp/tst_set
        
        
        
    def interpolate(self , tr_ds_clf , tr_labels_clf):
    
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
                low_bound = np.argmin([np.abs(x - v_i) for x in self.c_vec])
                
                if v_i < self.c_vec[low_bound] and low_bound <> 0:
                    low_bound -= 1
                    
                lows.append(low_bound)
                
            ds_sample["lows"] = lows
        
            ##find hypercube
            hypercube = []
            for l in lows:
                hypercube.append([self.c_vec[l] ,self.c_vec[l + 1]])
                    
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
        
        
        
    def main_grps_run(self):
        
        tr_grp = []
        te_grp = []
        
        grps_clf[0] = self.grps[0][:self.grps_len]
        grps_clf[1] = self.grps[1][:self.grps_len]
        for g1 in grps_clf[0]:
            g2 = grps_clf[1][grps_clf[0].index(g1)]
            ds_out_pairs = []
            
            #get group indices data
            ds_out_pairs =  []
            for d in self.ds:
                sample = [d[0][g1[0]] , d[0][g1[1]] , d[1][g2[0]] , d[1][g2[1]]]
                ds_out_pairs.append({"vec":sample , "label":d[2]})
                
            # separate to train-test
                
            te_set = ds_out_pairs[:int(self.test_pr*len(ds_out_pairs))]
            tr_set = ds_out_pairs[int(self.test_pr*len(ds_out_pairs)):]
        
        
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
                    
            if kmeans_flag:
                ds_temp =(np.array(tr_ds_clf).reshape(-1, 1))
                k_means = cluster.KMeans(self.C)
                k_means.fit(np.asarray(ds_temp))
                c_vec = (k_means.cluster_centers_)
                
                c_vec = sorted([cv[0] for cv in c_vec])
                c_vec.insert(0 , self.d_min - self.min_max_margin)
                c_vec.append(self.d_max + self.min_max_margin)
                
            self.c_vec = c_vec
            print "c vector genarated\n"   
        
        
            # interpolation ---------------------------------------------------------------
            
            print "interpolate.........."
            
            train_ds_json = self.interpolate(tr_ds_clf , tr_labels_clf)
            test_ds_json = self.interpolate(te_ds_clf , te_labels_clf)
            
            # embed -----------------------------------------------------------------------
            
            train_ds_json = embed(train_ds_json ,self.c_vec)
            test_ds_json = embed(test_ds_json ,self.c_vec)
            
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
        
        return 100*fp/tst_set
        
        
        
#####################################################################################################################        
        
# load dataset ----------------------------------------------------------------
        
with open("fn.json") as f:
    ds = f.readlines()

ds_new = []    
for d in ds:
    ds_new.append(json.loads(d.replace("'",'"')))

ds_new = 5*ds_new

# set groups for classification and pair matching tasks -----------------------
    
grps_clf1 = [(40, 44), (41, 45), (43, 44), (40, 47), (48, 52), (49, 53), (51, 52), (48, 55), (56, 60), (57, 61), (59, 60), (56, 63), (64, 68), (65, 69), (67, 68), (64, 71), (72, 76), (73, 77), (75, 76), (72, 79)]
grps_clf2 = [(42, 46), (43, 47), (42, 45), (41, 46), (50, 54), (51, 55), (50, 53), (49, 54), (58, 62), (59, 63), (58, 61), (57, 62), (66, 70), (67, 71), (66, 69), (65, 70), (74, 78), (75, 79), (74, 77), (73, 78)]
grps_clf = [ grps_clf1 , grps_clf1 ]


############# singles groups indices ################################################

grps_clf = []
for i in range(40,60,4):
    grps_clf.append((i,i+1,i+2,i+3))

###############################################################################

# labels indices

path_ims = "/home/yakir/idd/datasets/retargeting/"
path_ims = "C:\ofir_stuff_idd\datasets\dataset-20110512"
labels = sorted(os.listdir(path_ims))

C_list = range(2,12,2)
#g_len_list = range(4,len(grps_clf))
g_len_list = range(1,len(grps_clf))
#g_len_list = [1]
results = {}
for c_ind in C_list:
    results[c_ind] = []
    for g_len in g_len_list:

        id_pairs = ID(grps =  grps_clf , ds = ds_new , C = c_ind , grps_len = g_len)
        id_pairs.set_ext_vals()
        #id_pairs.set_pairs_ds()
        #fp = id_pairs.main_grps_run()
        
        fp = id_pairs.main_sngl_run()
        results[c_ind].append(fp)
        
        
        

