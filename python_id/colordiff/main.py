# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:17:49 2017
@author: yakir
"""

import json
from prepare_ds import prep_ds, de_pair
from edit_x_tr import arrange_x
from id_color_diff_cam import embed_main, arrange_ds
from train_test_sgd import train , test

def run_idd(cam = False, fw = True, c = ""):
    
    #select data - munswell / furnswarth
    if fw:
        import munswell as ds
    else:
        import furnsworth as ds
    # select experiment type - cam

    #load datasets
    refs = ds.refs
    sd = ds.sd ; nc = ds.nc ; km = ds.km ; kd = ds.kd
    prep_ds(refs  , km , kd , nc , sd , cam = cam , fw = fw)
    if cam:
        tr_fn = "rgb_set_tr_cam_furnsworth.json"
        te_fn = "rgb_set_te_cam_furnsworth.json"
    else:
        tr_fn = "rgb_set_tr_furnsworth.json"
        te_fn = "rgb_set_te_furnsworth.json"
        
    cvecs = embed_main(tr_fn , te_fn, c, cam = cam)
    X_tr , X_te , Y_tr , Y_te = arrange_x(cam = cam)
    # Calculate coefficients
    l_rate = 0.1
    n_epoch = 50
    w = train(X_tr,Y_tr, l_rate, n_epoch, weight = "", reg = "")
    errors = test(X_te, Y_te, w, method = "L2")
    print "cam = ", cam
    print "c = ", c
    return {"err":errors,"c":c}

if __name__ == "__main__":
    out = {"cam":{}, "color":{}}
    #for _ in range(10):
    for c in range(2,9):
        out["color"][c] = run_idd(cam = False, fw = True, c = c)
<<<<<<< HEAD
    for c in range(2,3):
=======
    for c in range(2,9):
>>>>>>> 9e89fafcff4033420920eb4eaba6d32ce40184a5
        out["cam"][c] = run_idd(cam = True, fw = True, c = c)
        