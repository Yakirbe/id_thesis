# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 17:09:55 2016
@author: yakir
"""
import numpy as np
import sys

class sp_emb_mat():
    def __init__(self , dns_mat = np.asarray([[]])):
        
        self.size = np.asarray(dns_mat).shape
        self.inds = []
        self.vals = []
        self.dense = []
        self.dns_inds = []
        self.merged_inds = []
        self.merged_dns = []
        
        for dns_vec in dns_mat:
            inds = []        
            vals = []
            i = 0
            for d in dns_vec:
                if d <> 0:
                    inds.append(i)
                    vals.append(d)
                i += 1
        
            self.inds.append(inds)
            self.vals.append(vals)
            
        self.remove_zeros()
            
    def print_sp(self):
        print "size = " ,  self.size
#        print "inds = " , self.inds
#        print "vals = " ,  self.vals
        
        
    def remove_zeros(self):
        
        print "removing zeros and assigning dense shape of matrix:"
        
        in_inds = []
        dense = []
        for v_ind in self.inds:
            in_inds.extend(v_ind)
            in_inds = list(set(in_inds))
            
        print "dense indices = {}".format(str(in_inds))            
        self.dns_inds = in_inds
        
        for v_ind in self.inds:
            dns_vec = [0.0]*len(in_inds)
            for i in in_inds:
                if i in v_ind:
                    dns_vec[in_inds.index(i)] = self.vals[self.inds.index(v_ind)][v_ind.index(i)]
            dense.append(dns_vec)
            
            self.dense = dense

        print "done!\n"

    def merge_mats(self , new_X):
        
        print "merging:"
        
        merged_inds = list(set(self.dns_inds+new_X.dns_inds))
        dense = []
        inds = []
        vals = []
        for v_ind in self.inds:
            inds.append(merged_inds)
            dns_vec = [0.0]*len(merged_inds)
            for i in merged_inds:
                if i in v_ind:
                    dns_vec[merged_inds.index(i)] = self.vals[self.inds.index(v_ind)][v_ind.index(i)]
            dense.append(dns_vec)
            vals.append(dns_vec)
            
        for v_ind in new_X.inds:
            inds.append(merged_inds)
            dns_vec = [0.0]*len(merged_inds)
            for i in merged_inds:
                if i in v_ind:
                    dns_vec[merged_inds.index(i)] = new_X.vals[new_X.inds.index(v_ind)][v_ind.index(i)]
            dense.append(dns_vec)
            vals.append(dns_vec)
            
            
        self.merged_inds = merged_inds            
        self.merged_dns = dense
        self.inds = inds        
        self.vals = vals
        print "merged set length = " , len(self.inds)        
        print "done!\n"        
        
        
        
        
        
        
        
if __name__ == "__main__":
        
    mat_sp_tr = sp_emb_mat(X_tr)
    mat_sp_te = sp_emb_mat(X_te)
    
    mat_sp_tr.merge_mats(mat_sp_te)
    mat_sp_tr.merge_mats(mat_sp_te)
    mat_sp_tr.merge_mats(mat_sp_tr)
    
    #mat_sp.remove_zeros()
