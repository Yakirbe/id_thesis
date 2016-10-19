"""
This module shows some examples of Delta E calculations of varying types.
"""

# Does some sys.path manipulation so we can run examples in-place.
# noinspection PyUnresolvedReferences
import example_config
from skimage import io, color
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_diff import delta_e_cie1976, delta_e_cie1994, \
    delta_e_cie2000, delta_e_cmc


def get_color_list(string):
    
    col_list = []
    refs_d = string.split("\n")[:-1]
    for r in refs_d:
        col_list.append([float(x) for x in r.split("\t")])
    return col_list



def get_rgb_label_set(kd_d  , refs_d , th = 5):
    
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
            if de_c2k <= th and j == im_num:
                tot +=1
                final_set.append({"v1":color1_srgb.get_value_tuple(),"v2":color2_srgb.get_value_tuple(),"dis":de_c2k})
                print "tot = {}".format(tot)
                print "rgb:", sRGBColor(kd_d[i][0] , kd_d[i][1] , kd_d[i][2])
                print "color1 lab:" , color1.get_value_tuple()
                print "color2 lab:" , color2.get_value_tuple()
                print j , im_num ,":"
                print "de_c2k: %.3f" % de_c2k , "\n"
                
    return final_set

# get ref table

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



