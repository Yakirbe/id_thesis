# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:17:49 2017

@author: yakir
"""
from prepare_ds import prep_ds
from edit_x_tr import arrange_x
from id_color_diff_cam import embed_main, arrange_ds
from train_n_pred import train_id , pred_id
if __name__ == "__main__":
    #select data - munswell / furnswarth
    fw = True
    if fw:
        import munswell as ds
    else:
        import furnsworth as ds
    # select experiment type - cam
    cam = True

    #load datasets
    #refs = ds.refs
    #sd = ds.sd ; nc = ds.nc ; km = ds.km ; kd = ds.kd
    #prep_ds(refs  , km , kd , nc , sd , cam = cam , fw = fw)
    #tr_fn = "rgb_set_tr_cam_furnsworth.json"
    #te_fn = "rgb_set_te_cam_furnsworth.json"
    #cvecs = embed_main(tr_fn , te_fn)
    X_tr , X_te , Y_tr , Y_te = arrange_x()
    #model = train_id(X_tr , Y_tr)
    #pred_id(model , X_te , Y_te)
