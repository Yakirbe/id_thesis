# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 01:05:29 2016

@author: yakir
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

#image = cv2.imread("butterfly.png")
#[h,w] = image.shape[:2]
#cv2.circle(image , (w/2,h/2) , np.min([w/2,h/2]) , (255,0,0) , 5)
#cv2.line(image , (w/2,h/2) , (w/2,0) , (255,0,0) , 5)
#cv2.circle(image , (w/2,h/2) , 10 , (0,0,0) , -1)
#cv2.imshow("butterfly",image)
#cv2.imwrite("butterfly_sift.png",image)


#with open("pairs_res.json","w") as f:
#    json.dump(results , f)

#vec_inds = range(128)
#neightbors = [57, 58, 59, 60, 61, 62, 63]
#ref = 60
#
#
#
#blks = []
#blk = []
#for i in vec_inds:
#    if (i+1) % 8 == 0 :#and i >0:
#        blk.append(i)
#        blks.append(blk)
#        print blk
#        blk = []
#    else:
#        blk.append(i)
#
#pairs_logics = [((0,4) , (2,6)) , ((1,5) , (3,7)) , 
#                ((3,4) , (2,5)) , ((0,7) , (1,6))]# , 
#                #((0,4) , (2,6)) , ((1,5) , (3,7))]
#                
#gr1 = []
#gr2 = []
#
#for b in blks[5:10]:
#    for p in pairs_logics:
#        
#        gr1.append((b[p[0][0]] , b[p[0][1]]))
#        gr2.append((b[p[1][0]] , b[p[1][1]]))
#    
#print gr1
#print gr2