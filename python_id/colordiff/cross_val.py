# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 23:37:08 2017

@author: yakir
"""

import random, itertools, math

set_x = [random.uniform(0, 10) for _ in range(1000)]

k = 5

def group(lst, div):
    lst = [ lst[i:i + len(lst)/div] for i in range(0, len(lst), len(lst)/div) ] #Subdivide list.
    if len(lst) > div: # If it is an uneven list.
        lst[div-1].extend(sum(lst[div:],[])) # Take the last part of the list and append it to the last equal division.
    return lst[:div] #Return the list up to that point.

l = group(set_x, 6)
g = l[3]
ll = [sub for sub in l if sub <> g]
merged = list(itertools.chain.from_iterable(ll))
