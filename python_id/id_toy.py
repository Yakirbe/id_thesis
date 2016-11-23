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
        label_num = labels.index(label)
        
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

ds_new = 5*ds_new
# set groups for classification and pair matching tasks -----------------------
    
grps_clf = []
for i in range(40,50,4):
    grps_clf.append((i,i+1,i+2,i+3))
    
        
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
    if random.uniform(0,1) <= test_pr:
        te_set.append(d)
    else:
        tr_set.append(d)
        
# shuffle
tr_set = random.sample(tr_set , len(tr_set))
te_set = random.sample(te_set , len(te_set))

# generate dataset (train and test sets)



#grp_ind = 0
tr_grp = []
te_grp = []

for grp_ind in range(len(grps_clf)):
    
    tr_ds_clf = []
    tr_labels_clf = []

    te_ds_clf = []
    te_labels_clf = []
    
    for i in range(len(tr_set)):
        samp_vec = tr_set[i]["vec"]
        tr_labels_clf.append(str(tr_set[i]["label"]))
        v = [samp_vec[s] for s in grps_clf[grp_ind]]
        tr_ds_clf.append(v)
        
        
    for i in range(len(te_set)):
        samp_vec = te_set[i]["vec"]
        te_labels_clf.append(str(te_set[i]["label"]))
        v = [samp_vec[s] for s in grps_clf[grp_ind]]
        te_ds_clf.append(v)


    # generate dis vecs
        
    C = 3   
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
        #c_vec[0] -= min_max_margin
        #c_vec[-1] += min_max_margin
        
    #else:
    #    self.c_max += self.min_max_margin
    #    self.c_min -= self.min_max_margin
    #    c_vec = list(np.linspace(self.c_min , self.c_max , self.C))
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

tr_grp_emb = [[]]*len(tr_grp[0])
tr_grp_label = [-1]*len(tr_grp[0])

for g in tr_grp:
    for i in range(len(tr_grp_emb)):
        tr_grp_emb[i] = tr_grp_emb[i] + list(g[i]["embedded"].toarray()[0])
        tr_grp_label[i] = g[i]["label"]
        
#test set

te_grp_emb = [[]]*len(te_grp[0])
te_grp_label = [-1]*len(te_grp[0])
        
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
#
#
#
#cons = test_ds_json[0]["label"]

#for s in test_ds_json:
#    
#    if s["label"] == cons:
#        print s["v_vec"] , s["coef_vec"] , s["label"]
















# pairing matching ------------------------------------------------------------






















































def helper(lst):
    lst1, lst2 = [], []
    for el in lst:
        lst1.append(el[0])
        lst2.append(el[1])
    return lst1, lst2
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


class idd(object):
    
    def __init__(self ,dataset = ds ,  ind_groups = [{} , {} , {}] , min_max_margin = 1 ,test_pr = 0.2 , kmeans_flag = True , C = 10 ):
        print """
        init idd session:
        
        dataset length = {}
        kmeans flag = {}
        C const = {}
        test percentage = {}
        
        """.format(len(dataset) , kmeans_flag , C , test_pr)
        
        self.ds_json = {}
        self.min_max_margin = min_max_margin
        self.ds_length = len(dataset)
        self.test_pr = test_pr
        self.kmeans_flag = kmeans_flag
        self.C = C
        self.c_vec = []
        self.c_mat = []
        self.train_set = {}
        self.test_set = {}
        self.c_max = -1
        self.c_min = 1000
        self.ds = []
        self.ind_groups = ind_groups
        
        
        
    def norm_0_1(self , val , m_min , m_max):
        """
        this function normalizes between 0 - 1
        """
        if val == m_min:
            return 0.0
        elif val == m_max:
            return 1.0
        else:
            return (val - m_min)/(m_max - m_min)
            
            
            
    def norm_vec(self , v_vec , cube):
        n_vec = []
        for v in v_vec:
            n_vec.append(self.norm_0_1(v , cube[v_vec.index(v)][0] , cube[v_vec.index(v)][1]))  
        return n_vec
        
        
          
    def using_indexed_assignment(self , x):
        result = np.empty(len(x), dtype=int)
        temp = np.argsort(x)
        result[temp] = np.arange(len(x))
        result = [np.max(result) - r for r in result]
        return list(result)
        
        
        
    def find_simplex(self , v_vec):
        args = self.using_indexed_assignment(v_vec)
        refs = []
        for r in range(len(v_vec) + 1):
            ref = [0]*len(v_vec)
            
            for ar in args:
                if r > ar:
                    ref[args.index(ar)] = 1
            refs.append(ref)
        return refs
        
        
    def find_coef_vec(self , refs , v_vec):
     
        # Now simply solve for x
        coef_vec = np.tensordot((np.asarray(refs[1:])), v_vec[1:]) 
        
        return coef_vec
        
        
    def prepare_toy_dataset(self):
        
        for i in range(self.ds_length):
            
            ter_sample = True
            if random.uniform(0,1) <= self.test_pr:
                ter_sample = False
            
            dataset_labeled[i] = {"label":label , "v1":v1 , "v2":v2 , "train":ter_sample, "x_grp":[v1_l , v2_l]}
            if ter_sample:
                self.ds.extend(v1)
                self.ds.extend(v2)
            #del v1 , v2
            
        self.ds_json = dataset_labeled
        
        print "dataset prepared\n"
        
        
    def gen_c_vec(self):
        
        print "genarating c vector....."
                   
        if self.kmeans_flag:
            self.ds =(np.array(self.ds).reshape(-1, 1))
            k_means = cluster.KMeans(self.C)
            k_means.fit(np.asarray(self.ds))
            c_vec = (k_means.cluster_centers_)
            
            c_vec = sorted([cv[0] for cv in c_vec])
            c_vec[0] -= self.min_max_margin
            c_vec[-1] += self.min_max_margin
    #        c_vec.insert(0,c_vec[0] - 1)
    #        c_vec.append(c_vec[-1] + 1)
        else:
            self.c_max += self.min_max_margin
            self.c_min -= self.min_max_margin
            c_vec = list(np.linspace(self.c_min , self.c_max , self.C))
        print "c vector genarated\n"    
        
        self.c_vec = c_vec
        self.c_mat = np.array([self.c_vec , self.c_vec])


#our objective is to find the distance between 2 2d vectors,
#and to prove its co-exist the dataset behavior

    def interpolate(self):
        
        print "interpolate.........."
        
        for samp in self.ds_json:
        
            v1 = self.ds_json[samp]["v1"]
            v2 = self.ds_json[samp]["v2"]
            
            x0 = v1[0]
            x1 = v1[1]
            y0 = v2[0]
            y1 = v2[1]
            
            v_vec = [x0 , x1 , y0 , y1]
            lows = []
            for v_i in v_vec:
                low_bound = np.argmin([np.abs(x - v_i) for x in self.c_vec])
                
                if v_i < self.c_vec[low_bound] and low_bound <> 0:
                    low_bound -= 1
                
                    
                lows.append(low_bound)
            self.ds_json[samp]["lows"] = lows
            ##find hypercube

            self.ds_json[samp]["cube"] = [[self.c_vec[lows[0]] ,self.c_vec[lows[0] + 1]] ,
                    [self.c_vec[lows[1]] ,self.c_vec[lows[1] + 1]] ,
                    [self.c_vec[lows[2]] ,self.c_vec[lows[2] + 1]] ,
                    [self.c_vec[lows[3]] ,self.c_vec[lows[3] + 1]]]
            #dataset_labeled[samp]["cube"] = cube
                                            
            ##normalize
            self.ds_json[samp]["n_vec"] = self.norm_vec(v_vec , self.ds_json[samp]["cube"])
        
            ##find simplex
            self.ds_json[samp]["refs"] = self.find_simplex(self.ds_json[samp]["n_vec"])
            ##calc coefficients    
            coef_vec = list(np.linalg.solve(np.transpose(np.asarray(self.ds_json[samp]["refs"][1:])) , self.ds_json[samp]["n_vec"]))
            coef_vec.insert( 0 , 1 - sum(coef_vec) )
            
            self.ds_json[samp]["coef_vec"] = coef_vec
            
        print "dataset interpolated\n"
    
    
    def embed(self):
        
        print "embed..."
        
        for samp in self.ds_json:

            emb_vec = [0.0]*self.c_mat.shape[1]**(2*self.c_mat.shape[0])
            
            refs_bounded = []
            
            for r in self.ds_json[samp]["refs"]:
                refs_bounded.append(map(add, r, self.ds_json[samp]["lows"]))
            for j in range(len(refs_bounded)):
                ind = 0
                for i in range(len(refs_bounded[0])-1): 
                    ind += self.C*refs_bounded[j][i]
                ind += refs_bounded[j][-1]
                
                emb_vec[ind] = self.ds_json[samp]["coef_vec"][j]
                
            emb_vec = csr_matrix(emb_vec , dtype=np.float64)
            self.ds_json[samp]["embedded"] = emb_vec
                
        print "dataset embedded\n"
        
        
    def train(self):
        
        print "training..."
#        for samp in self.ds_json :
#            if self.ds_json[samp]["train"]:
#                print list(10*self.ds_json[samp]["embedded"].toarray()[0])
#                print np.sum(10*self.ds_json[samp]["embedded"].toarray()[0]) , len(10*self.ds_json[samp]["embedded"].toarray()[0])
        X_tr = [list(self.ds_json[samp]["embedded"].toarray()[0]) for samp in self.ds_json if self.ds_json[samp]["train"]]
        Y_tr = [self.ds_json[samp]["label"] for samp in self.ds_json if self.ds_json[samp]["train"]]
        
        #self.clf = svm.SVC(C = 1e5 , decision_function_shape='ovr' , kernel='rbf' , cache_size  = 200)
        self.clf = svm.SVR(C = 1e5 , kernel='rbf' , cache_size  = 200)
        self.clf.fit(X_tr, Y_tr) 
        
        print "training done\n"
        
    
    def test(self , var_i):
        
        print "testing..."
        
        X_te = [list(self.ds_json[samp]["embedded"].toarray()[0]) for samp in self.ds_json if not self.ds_json[samp]["train"]]
        Y_te = [self.ds_json[samp]["label"] for samp in self.ds_json if not self.ds_json[samp]["train"]]
        
        tst_set = len(X_te)
        tp = 0.0
        fp = 0.0
        for x in X_te:
            if X_te.index(x)%100 == 0:
                print  X_te.index(x) , "done from" , len(X_te)
                print  Y_te[X_te.index(x)] , self.clf.predict([x])[0]
                
            if np.abs(Y_te[X_te.index(x)] - self.clf.predict([x])[0]) <= 2*var_i:
                tp += 1
            else:
                fp += 1
        print "true positive = " , tp , "of" , tst_set
        print "false positive = " , fp , "of" , tst_set
        print "tp % = " , 100*tp/tst_set
        print "fp % = " , 100*fp/tst_set
        print "testing done\n\n\n\n"
        
        return 100*fp/tst_set
        
        
        
    def plot_ds(self):
        
        ds = []
        n = []
        for samp in self.ds_json:
            ds.append(self.ds_json[samp]["v1"])
            n.append(self.ds_json[samp]["x_grp"][0])
            ds.append(self.ds_json[samp]["v2"])
            n.append(self.ds_json[samp]["x_grp"][1])
            
        x_p , y_p = helper(ds)
        
        fig, ax = plt.subplots()
        ax.scatter(x_p, y_p)
        
        for i, txt in enumerate(n):
            ax.annotate(txt, (x_p[i],y_p[i]))
            plt.plot(x_p , y_p , "ro")

def main_idd():
        flags = [True , False]
        fl_arr = {}
        for fl in flags:
            
            var_i = 0.1
            var_end = 1.0
            #var_end = 0.1
            var_step = 0.1
            
            print "kmeans?" , fl
            
            var_arr = []
            var_x_arr = []
            while var_i <= var_end:
                
                C_i = 2
                C_end = 8
                #C_end = 4
                C_step = 1
                
                print "data variance = " , var_i
                var_x_arr.append(var_i)
                fp_arr = []
                C_arr = []
                
                while C_i <= C_end:
                    
                    print "C vector length = " , C_i
                    t0 = time.time()
                    idd_sess = idd(C = C_i , kmeans_flag = fl , amp = var_i)
                    C_arr.append(C_i)
                    idd_sess.prepare_toy_dataset()
                    idd_sess.gen_c_vec()
                    print "c vector =", idd_sess.c_vec
                    idd_sess.interpolate()
                    idd_sess.embed()
                    idd_sess.train()
                    fp_arr.append(idd_sess.test(var_i))
                    #idd_sess.plot_ds()
                    print "process time = ", "%.2f" % (time.time() - t0) , "sec\n\n\n\n"
                    C_i += C_step
                    
                var_i += var_step
                var_arr.append(fp_arr)
                
            fl_arr[str(fl) + "kmeans"] = var_arr
            
            
        del idd_sess
        
        with open('data.json', 'w') as outfile:
            json.dump(fl_arr, outfile)
            

        


            
def main_idd2():
        idd_sess = idd(C = 8 , kmeans_flag = True , amp = 1.5 , ds_length = 500)
        idd_sess.prepare_toy_dataset()
        ds = []
        for el in idd_sess.ds_json.keys():
            print idd_sess.ds_json[el]["v1"]
            print idd_sess.ds_json[el]["v2"]
            ds.append(idd_sess.ds_json[el]["v1"])
            ds.append(idd_sess.ds_json[el]["v2"])
            plt.plot(idd_sess.ds_json[el]["v1"][0] , idd_sess.ds_json[el]["v1"][1] , "yo")
            plt.plot(idd_sess.ds_json[el]["v2"][0] , idd_sess.ds_json[el]["v2"][1] , "ro")
            
        


if __name__ == "__maindd__":
    
    main_idd()
    
    labels = []
    plotHandles = []


    var_i = 0.1
    var_end = 1.1
    #var_end = 0.1
    var_step = 0.1
    var_vals = np.arange(var_i , var_end , var_step )
    
            
    C_i = 2
    C_end = 16
    #C_end = 4
    C_step = 2
    c_vals = np.arange(C_i , C_end , C_step )
    
    with open('data.json') as outfile:
           dataset = json.load(outfile)
           
    for t_f in dataset.keys():
        fig = plt.figure()
        print "\n\n" , t_f  ,":"
        i_val = 0
        for vec in dataset[str(t_f)]:
            var_val = var_vals[i_val]
            #if var_val%0.05 == 0:
            if True:
                print var_vals[i_val] , vec
                x1, = plt.plot(c_vals, vec) #need the ',' per ** below
                plotHandles.append(x1)
                labels.append(var_vals[i_val])
                x2, = plt.plot(c_vals, vec , "ko") #need the ',' per ** below
                #plotHandles.append(x2)
                #labels.append(var_vals[i_val])
            
            i_val += 1
        plt.title(t_f)
        plt.xlabel("C vector length")
        plt.ylabel("test error percentage")
        my_xticks = c_vals
        plt.xticks(np.arange(2,14,2), my_xticks)
        plt.legend(plotHandles, labels, 'upper left',ncol=1)
        


