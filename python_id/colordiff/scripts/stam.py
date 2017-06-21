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
    
                
                
#%% test vs train errors
print "test vs train errors"
cs = range(2,11)
err = "err_te"
cam = "color"
reg = "l1"

p1 = []
for c in cs:
    if err == "err_te":
        p1.append(jd[reg][cam][str(c)][err][0])
    else:
        p1.append(jd[reg][cam][str(c)][err][1])

err = "err_tr"
p2 = []
for c in cs:
    if err == "err_te":
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


t = jd[reg][cam][str(c)]["c"]
plt.imshow(t)#, interpolation='nearest')
plt.show()




#%% regs compare

print "regs compare"
err = "err_te"
reg = "all"

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

reg = "de"
p3 = []
for c in cs:
    if reg == "de":
        p3.append(jd[reg][cam][str(c)][err][0])
    else:
        p3.append(jd[reg][cam][str(c)][err][1])


fig = plt.figure(figsize=(16,9))

plt.ylabel('error')
plt.xlabel('centers per dimension')

line1, = plt.plot(cs,p1,linewidth=2,marker="o", label='L1')
line2, = plt.plot(cs,p2, linewidth=2,marker="o", label='L2')
line3, = plt.plot(cs,p3, linewidth=2,marker="o", label='delta_e')
plt.legend(handles=[line1,line2,line3], loc=1)
plt.show()

t = jd[reg][cam][str(c)]["c"]
plt.imshow(t)#, interpolation='nearest')
plt.show()


#%% cam vs color
print "cam vs color"

cs = range(2,11)
err = "err_te"
cam = "color"
reg = "l1"

p1 = []
for c in cs:
    if cam == "color":
        p1.append(jd[reg][cam][str(c)][err][0])
    else:
        p1.append(jd[reg][cam][str(c)][err][1])

cam = "color"
p2 = []
for c in cs:
    if cam == "cam":
        p2.append(jd[reg][cam][str(c)][err][0])
    else:
        p2.append(jd[reg][cam][str(c)][err][1])

fig = plt.figure(figsize=(16,9))

plt.ylabel('error')
plt.xlabel('centers per dimension')

line1, = plt.plot(cs,p1,linewidth=2,marker="o", label='homogeneous division')
line2, = plt.plot(cs,p2, linewidth=2,marker="o", label='camera based division')
plt.legend(handles=[line1,line2], loc=1)
plt.show()


t = jd[reg][cam][str(c)]["c"]
plt.imshow(t)#, interpolation='nearest')
plt.show()


