# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 11:01:45 2016

@author: yakir
"""
import json
import scipy.io
mat = scipy.io.loadmat('../../datasets/retargeting/ref_results/subjData-blind_37.mat')

labels = [str(x[0][0]) for x in mat["subjData"][0][0][0]]
labels = [l.split("_0.")[0] for l in labels]
vecs = [list(x) for x in mat["subjData"][0][0][1]]

#CR SV MULTIOP SC SCL SM SNS WARP
op_list = ["_cr","_multiop","_sc","_scl","_sm","_sns","_sv","_warp"]
op_order = [1,10,3,6,7,8,9,11]
op_list_non = ["_lg","_osa","_qp"]

with open("fn.json" ) as f:
    res = f.read().split("\n")
del res[-1]

data = {}
for l in labels:
    data[l] = {"usr":vecs[labels.index(l)] , "ref":[] ,
         "ref_8":{} , "D":{}}
    for p in op_list:
        data[l]["ref_8"][p] = []
        data[l]["D"][p] = -1.0
    
refs = []
i_cur = ""
for i in im_list_short:
    ind = im_list_short.index(i)
    if all([op not in i for op in op_list + op_list_non]):# and ([op not in i for op in op_list_non]):
        rr = i.split(".")[0]
        print "\n"
        count = 0
        
        for l in labels:
            #if rr in l and rr not in refs and rr not in ["ski" , "car" , "surfer"]:
            if rr == l :
                print rr , l
                refs.append(rr)
                data[l]["ref"] = res[ind].split("], 'label'")[0].split("vec': [")[1].split(", ")
                data[l]["ref"] = [float(x) for x in data[l]["ref"]]
                for op in op_list:
                    print op , im_list_short[ind + op_order[op_list.index(op)]]
                    data[l]["ref_8"][op] = res[ind + op_order[op_list.index(op)]].split("], 'label'")[0].split("vec': [")[1].split(", ")
                    data[l]["ref_8"][op] = [float(x) for x in data[l]["ref_8"][op]]
                    print count , l , i , rr , op
                    count += 1
        

