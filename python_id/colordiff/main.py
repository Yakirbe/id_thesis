# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:17:49 2017

@author: yakir
"""
from prepare_ds import prep_ds
from edit_x_tr import arrange_x
from id_color_diff_cam import embed_main, arrange_ds
from train_test_sgd import train , test

def run_idd(cam = False, fw = True, c = ""):
    
    #select data - munswell / furnswarth
    fw = True
    if fw:
        import munswell as ds
    else:
        import furnsworth as ds
    # select experiment type - cam
    cam = False

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
        
    cvecs = embed_main(tr_fn , te_fn, c)
    X_tr , X_te , Y_tr , Y_te = arrange_x()
    # Calculate coefficients
    l_rate = 0.01
    n_epoch = 5000
    w = train(X_tr,Y_tr, l_rate, n_epoch, weight = "")
    l2_error = test(X_te, Y_te, w,method = "L1")
    print "cam = ", cam
    print "c = ", c

if __name__ == "__main__":
    for c in range(2,6):
        run_idd(cam = False, fw = True, c = c)
    for c in range(2,6):
        run_idd(cam = True, fw = True, c = c)
    
