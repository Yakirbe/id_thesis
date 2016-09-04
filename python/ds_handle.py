# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 11:42:43 2016

@author: yakir
"""
import os , fnmatch
import json
import pickle
import numpy as np
import cv2
from matplotlib import pyplot as plt

from scipy.spatial.distance import euclidean


def get_mid_sift(img , sift):
    
    kp_x = img.shape[1]/2
    kp_y = img.shape[0]/2
    kp_size = np.min((kp_x , kp_y))
    kp_mid = [cv2.KeyPoint(x = kp_x , y = kp_y , _size = kp_size , _angle = 0)]
    
    # find the descriptor with SIFT
    des_mid = sift.compute(img1,kp_mid , None)
    return list(des_mid[1][0])
    
    
    
def match_mid_imgs(img1 , img2 , sift):
    
    dst = euclidean(get_mid_sift(img1 , sift),get_mid_sift(img2 , sift))
    print "images distance = " , dst
    return dst
    
    
    
def find_files(dis , pat):
    
    for root , dirs , files in os.walk(dis):
        for basename in files:
            if fnmatch.fnmatch(basename , pat):
                filename = os.path.join(root , basename)
                yield filename
              
              
def find_files_list(dis , pat):
    return list(sorted(find_files(dis , pat)))         
    


# Initiate SIFT detector
sift = cv2.SIFT()    
    
path_ims = "/home/yakir/idd/datasets/retargeting/"
im_list = find_files_list(path_ims , "*.png")
    
labels = sorted(os.listdir(path_ims))
ds = []

for im_path in im_list:
    print im_list.index(im_path) , im_path
    img1 = cv2.imread(im_path,0) # queryImage
    ds.append({"vec":get_mid_sift(img1 , sift) , "label":im_path.split("/")[-2]})


with open("fn.json" , "w")  as f:
    f.writelines("%s\n" % item for item in ds )
