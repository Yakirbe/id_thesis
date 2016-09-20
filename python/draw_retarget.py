# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 18:11:14 2016

@author: yakir
"""
import csv

#CR SV MULTIOP SC SCL SM SNS WARP
op_list = ["_cr","_sv","_multiop","_sc","_scl","_sm","_sns","_warp"]
#op_order = [1,10,3,6,7,8,9,11]
n = len(op_list)
#set pairs of op labels and d values:

k_list = [3,8]
k = 8
for d in data:
    usr_pairs = []
    id_pairs = []
    print d , ":"
    
    for op in op_list:
        ind = op_list.index(op)
        
        usr_pairs.append([op , data[d]["usr"][ind]])
        id_pairs.append([op , data[d]["D"][op]])
        
    usr_pairs.sort(key=lambda x: x[1])
    id_pairs.sort(key=lambda x: x[1])
    id_pairs.reverse()
    
    usr_pairs = usr_pairs[:(k+1)]
    id_pairs = id_pairs[:(k+1)]
    
    data[d]["id_pairs"] = id_pairs
    data[d]["usr_pairs"] = usr_pairs
    
    tau, p_val = scipy.stats.kendalltau(usr_pairs, id_pairs)
    data[d]["p_val"] = p_val
    data[d]["tau"] = tau
    print "tau = ", tau , " p value = ", p_val ,"\n"

################################################################################

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
    
    
    
###############################################################################

# plot
plt.close("all")  
    
N = len(att_dict)
taus = [att_dict[a]["tau_avg"] for a in att_dict]

mean = np.mean(taus)
std = np.std(taus)

print "taus =" , taus
print "mean =", mean
print "std =", std
ticks = tuple([a.replace("\\","\\n") for a in att_dict])

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, taus, width, color='r')

# add some text for labels, title and axes ticks
ax.set_ylabel('avg taus')
ax.set_title('average tau values')
ax.set_xticks(ind + 0.175)
ax.set_xticklabels(ticks , fontsize=14)

#ax.legend([rects1] , ['tau'])
ax.set_xlim(-1*width , (N)*3*width - 0.3)
ax.set_ylim(0.0 , np.max(taus)+0.1)

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                '%.2f' % (height), fontsize=20,
                ha='center', va='bottom')

autolabel(rects1)
plt.show()