# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:08:06 2016
@author: yakir
"""
import json
import os
from id_colors_diff import *
from sparse_embedded import sp_emb_mat


def get_max_l(X_tr):
    max_len = -1
    for el in X_tr:
        if max_len < len(el):
            max_len = len(el)
    return max_len


def zero_pad(X_tr , max_len):
            
    x_out = []
    for el in X_tr:
        x_out.append(el)
        if len(x_out[-1]) < max_len:
            x_out[-1] = x_out[-1] + [0.0]*(max_len - len(x_out[-1]) )
    return x_out
    

def arrange_x(cam = False):
    if cam:
        relevant_path = "../sets_jsons_cam/"
    else:
        relevant_path = "../sets_jsons/"
    included_extenstions = ['json']
    file_names = sorted([fn for fn in os.listdir(relevant_path)
                  if any(fn.endswith(ext) for ext in included_extenstions)])
    
    fn0 = relevant_path + file_names[0]
    
    with open(fn0) as f:
        data = json.load(f)
    X_tr = data["X_tr"]
    X_te = data["X_te"]
    Y_tr = data["Y_tr"]
    Y_te = data["Y_te"]
    for fn in file_names[1:]:
        #print relevant_path + fn
        #print len(X_tr[0])
        
        with open( relevant_path + fn) as f:
            data = json.load(f)
            
        X_tr.extend(data["X_tr"])
        X_te.extend(data["X_te"])
        Y_tr.extend(data["Y_tr"])
        Y_te.extend(data["Y_te"])
    max_l_tr = get_max_l(X_tr)
    max_l_te = get_max_l(X_te)
    m = max(max_l_te , max_l_tr)
    X_tr = zero_pad(X_tr, m)
    X_te = zero_pad(X_te, m)
    
    return X_tr , X_te , Y_tr , Y_te