# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 01:05:29 2016

@author: yakir
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("butterfly.png")
[h,w] = image.shape[:2]
cv2.circle(image , (w/2,h/2) , np.min([w/2,h/2]) , (255,0,0) , 5)
cv2.line(image , (w/2,h/2) , (w/2,0) , (255,0,0) , 5)
cv2.circle(image , (w/2,h/2) , 10 , (0,0,0) , -1)
cv2.imshow("butterfly",image)
cv2.imwrite("butterfly_sift.png",image)

