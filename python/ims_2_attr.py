# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 12:29:22 2016
@author: yakir
"""

import csv

with open('im_attr.csv', 'rb') as f:
    reader = list(csv.reader(f))

att_list = []    
for r in reader:
    
    if r <> ['', '', '', '', '', '', '', '']:
        att_list.append(r)
att_list = att_list[2:-2]
attrs = reader[0][2:]

att_dict = {}
for at in attrs:
    att_dict[at] = {}
    att_dict[at]["imlist"] = []
    att_dict[at]["tau_avg"] = 0.0
    #print attrs
for a in att_list:
    lbl = a[1]
    i = 0
    for x in a[2:]:
        if x:
            att_dict[attrs[i]]["imlist"].append(lbl)
        i += 1
        
for a in att_dict:
    at_avg = 0.0
    print "\n" , a
    for im in att_dict[a]["imlist"]:
        at_avg += data[im]["tau"]
    print at_avg/len(att_dict[a]["imlist"])
    att_dict[a]["tau_avg"] = at_avg/len(att_dict[a]["imlist"])
