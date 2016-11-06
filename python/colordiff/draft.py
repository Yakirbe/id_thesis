# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:08:06 2016

@author: yakir
"""

a =  [[0 , 0 , 0, 0, 0, 1],
         [0, 0, 1, 0, 0 ,1],
         [1, 0, 1, 0, 0, 1],
         [1, 1, 1, 0, 0, 1],
         [1, 1, 1, 1, 0, 1],
         [1, 1, 1, 1, 1, 1]] 
a = np.transpose(np.asarray(a))
 
b = [0.99276409848191216, 0.9718837470082502, 0.99783719484171696,
     0.46196357914891867, 0.08197911722536394, 0.99783719484171696] 
     
x = np.linalg.solve(np.transpose(np.asarray(mat)) , b)   
print a
print np.linalg.solve(np.transpose(np.asarray(mat)) , b)

print np.allclose(np.dot(a, x), b)
