# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:22:51 2017

@author: yakir
"""

import ast, os, json
import matplotlib.pyplot as plt
j_fn = "/home/yakir/Downloads/results.json"
with open(j_fn) as f:
    jd = json.loads(f.read())
    
    
for l in jd.keys():
    print l
    for field in jd[l]:
        print field
        for ma in sorted(jd[l][field]):
            print ma
            for f in jd[l][field][ma]:
                print f
                print jd[l][field][ma][f]
            print
                
        print 
    print "-------------------------------------------------------------------"
                
                
#%% test vs train errors

cs = range(2,11)
err = "err_te"
cam = "color"
reg = "l1"

p1 = []
for c in cs:
    if reg == "l1":
        p1.append(jd[reg][cam][str(c)][err][0])
    else:
        p1.append(jd[reg][cam][str(c)][err][1])


reg = "l2"

p2 = []
for c in cs:
    if reg == "l1":
        p2.append(jd[reg][cam][str(c)][err][0])
    else:
        p2.append(jd[reg][cam][str(c)][err][1])

fig = plt.figure(figsize=(16,9))

plt.ylabel('error')
plt.xlabel('centers per dimension')

line1, = plt.plot(cs,p1,linewidth=2,marker="o", label='test err')
line2, = plt.plot(cs,p2, linewidth=2,marker="o", label='train err')
plt.legend(handles=[line1,line2], loc=1)
plt.show()

#%% cam vs color

c = "10"
err = "err_te"
reg = "all"


#%% centers charts

c = "all"
err = "err_te"
reg = "all"
cam = "color"


#%% regs compare

c = "10"
err = "err_te"
reg = "all"
cam = "color"