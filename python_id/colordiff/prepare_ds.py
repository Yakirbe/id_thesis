"""
This module shows some examples of Delta E calculations of varying types.
"""

# Does some sys.path manipulation so we can run examples in-place.
# noinspection PyUnresolvedReferences
#import example_config
from skimage import io, color
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_diff import delta_e_cie1976, delta_e_cie1994, \
    delta_e_cie2000, delta_e_cmc
import random
import json

def get_color_list(string):
    
    col_list = []
    refs_d = string.split("\n")[:-1]
    for r in refs_d:
        col_list.append([float(x) for x in r.split("\t")])
    return col_list



def get_rgb_label_set(kd_d  , refs_d , th = 4, ds_size = 100):
    
    final_set = []
    # Color to be compared to the reference.
    tot = 0
    for i in range(len(kd_d)):
        
        color2_srgb = sRGBColor(kd_d[i][0]/255 , kd_d[i][1]/255 , kd_d[i][2]/255)
        color2 = convert_color(color2_srgb, LabColor, illuminant='D65')
        for j in range(len(refs_d)):
            # Reference color.
            color1 = LabColor(lab_l = refs_d[j][0] , lab_a = refs_d[j][1] , lab_b = refs_d[j][2])
            color1_srgb = convert_color(color1, sRGBColor, illuminant='D65')
            de_c2k =  delta_e_cie2000(color1, color2)
            im_num = i%len(refs_d)
            if de_c2k <= th:# and j == im_num:
                tot +=1
                v1 = [int(255*el) for el in color1_srgb.get_value_tuple()]
                v2 = [int(255*el) for el in color2_srgb.get_value_tuple()]
                final_set.append({"v1":v1,"v2":v2,"dis":de_c2k})
                if tot %100 == 0:
#                    print de_c2k
#                    print "color1 lab:" , color1_srgb.get_value_tuple()
#                    print "color2 lab:" , color2_srgb.get_value_tuple()
                    print "total ds = {}".format(tot)
                    
#                print "rgb:", sRGBColor(kd_d[i][0] , kd_d[i][1] , kd_d[i][2])
                
#                print j , im_num ,":"
#                print "de_c2k: %.3f" % de_c2k , "\n"
                if tot == ds_size:
                    print "got", ds_size, "samples"
                    return final_set



def prep_ds(refs  , km , kd , nc , sd , cam = False , fw = True):
    # get ref table
    print "preparing dataset from files,", cam, fw
    refs_d = get_color_list(refs)
        
    #get compare tables
        
    km_d = get_color_list(km)
    kd_d = get_color_list(kd)
    nc_d = get_color_list(nc)
    sd_d = get_color_list(sd)
    
    fl_km = get_rgb_label_set(km_d , refs_d)
    fl_kd = get_rgb_label_set(kd_d , refs_d)
    fl_nc = get_rgb_label_set(nc_d , refs_d)
    fl_sd= get_rgb_label_set(sd_d , refs_d)
    
    if cam:
        print "cam"
        if fw:
            print "fw"
            fn_out = "rgb_set_{}_cam_furnsworth.json"
        else:
            fn_out = "rgb_set_{}_cam_munswell.json"
    
        #camera ds prep    
        final_set_tr = fl_km + fl_kd + fl_nc# + fl_sd    
        final_set_te = fl_sd    
    else:
        print "no cam"
        if fw:
            fn_out = "rgb_set_{}_furnsworth.json"
        else:
            print "no fw"
            fn_out = "rgb_set_{}_munswell.json"
    
        #camera ds prep    
        final_set = fl_km + fl_kd + fl_nc + fl_sd    
        final_set = random.sample(final_set , len(final_set))
        final_set_tr = final_set[:int(0.8*len(final_set))]
        final_set_te = final_set[int(0.8*len(final_set)):]
        del final_set
    
    #regular ds prep    
    print "writing 2 files:{}",fn_out
    
    with open(fn_out.format("tr") , "w") as f:
        for samp in final_set_tr:
            json.dump(samp,f)#f.write(samp)
            f.write("\n")
        
    with open(fn_out.format("te") , "w") as f:
        for samp in final_set_te:
            json.dump(samp,f)#f.write(samp)
            f.write("\n")

if __name__ == "__main__":
    import munswell as mw
    refs = mw.refs
    sd = mw.sd ; nc = mw.nc ; km = mw.km ; kd = mw.kd
    prep_ds(refs  , km , kd , nc , sd)
