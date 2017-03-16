# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:22:51 2017

@author: yakir
"""

import ast, os, cv2, json
import shutil
import subprocess
class spot(object):

    def __init__(self , bbox = [] , cls = "" , start = "" , end = "" , duration = "",
                 size = [] ,position = [] , confidence = [] , frames = [] ,
                th_position = 0.5 , th_size = 0.5 , min_spot_wait = 2 , im_w = "" , im_h = "" , fps = "10"):
        self.bbox = bbox
        self.cls = cls
        self.start = start
        self.end = end
        self.frames = frames
        self.duration = duration
        self.size = size
        self.position = position
        self.confidence = confidence
        self.ended = False
        self.th_position = th_position
        self.th_size = th_size
        self.min_spot_wait = min_spot_wait
        self.im_w = im_w
        self.im_h = im_h
        self.fps = fps
        self.id = -1

    def init_from_dict(self ,dict_sp):


        dict_sp = ast.literal_eval(dict_sp)

        self.bbox = dict_sp["bbox"]
        self.cls = dict_sp["cls"]
        self.start = dict_sp["start"]
        self.end = dict_sp["end"]
        self.frames = dict_sp["frames"]
        self.duration = dict_sp["duration"]
        self.size = dict_sp["size"]
        self.position = dict_sp["position"]
        self.confidence = dict_sp["confidence"]
        self.ended = dict_sp["ended"]
        self.th_position = dict_sp["th_position"]
        self.th_size = dict_sp["th_size"]
        self.min_spot_wait = dict_sp["min_spot_wait"]
        self.im_w = dict_sp["im_w"]
        self.im_h = dict_sp["im_h"]
        self.fps = dict_sp["fps"]

    def print_sp(self):
        for s in self.__dict__:
            print s ,  self.__dict__[s]


    def test_fro_spot(self , fro):
        """
        this function checks whether a fro (mark from a frame)
        fits to the given spot
        """
        t_fro = fro["time"]
        if self.end:
            t_self = self.end
        else:
            t_self = self.start

        
        t_fro_int = int(t_fro.split(":")[0])*3600 + int(t_fro.split(":")[1])*60 + int(float(t_fro.split(":")[2]))
        t_self_int = int(t_self.split(":")[0])*3600 + int(t_self.split(":")[1])*60 + int(float(t_self.split(":")[2]))


        if False: #print conditions
            print self.cls , self.start , self.end
            print 'str(self.cls) == str(fro["cls"])' , str(self.cls) == str(fro["cls"])
            print 'L2(self.position[0] , fro["center"])= self.th_position' , self.position[-1] , fro["position"], L2(self.position[-1] , fro["position"]) <= self.th_position
            print 'np.abs(self.size[0] - fro["size"]) <= self.th_size' ,self.size[-1] ,fro["size"] ,  np.abs(self.size[-1] - fro["size"]) <= self.th_size
            print 'not self.ended' , not self.ended
            print "np.abs(t_fro_int - t_self_int) <= self.min_spot_wait)" , t_fro_int ,t_self_int , self.min_spot_wait , "\n"
        return (str(self.cls) == str(fro["cls"]) and
        
        L2_spt(self.position[-1] , fro["position"]) <= self.th_position and
        #L2(self.bbox[0] , fro["bbox"]) <= self.th_position and
        np.abs(self.size[-1] - fro["size"]) <= self.th_size and
        not self.ended and
        np.abs(t_fro_int - t_self_int) <= self.min_spot_wait)

    def assign_fro(self, fro):
        self.bbox.append(fro["bbox"])
        self.size.append(fro["size"])
        self.position.append(fro["position"])
        self.confidence.append(fro["score"])
        self.frames.append(fro["frame"])
        self.end = fro["time"]

def read_spots_from_file(path_f):

    #print "reading from file"

    if not os.path.exists(os.path.dirname(path_f)):
                os.makedirs(os.path.dirname(path_f))

    with open(path_f) as f:
        sp_strs = f.readlines()
        sp_set_strs = sorted(list(set(sp_strs)))
        spots_pool = []
        for sp_str in sp_set_strs:
            s = spot()
            sp_str = sp_str.replace("\n","").replace("'" , '"')
            s.init_from_dict(sp_str)
            spots_pool.append(s)

    return spots_pool


if __name__ == "__main__":    
    sp_fn = "/home/yakir/hs.out"
    sp_list = read_spots_from_file(sp_fn)
    
    for s in sp_list[:10]:
        s.print_sp()
        print
    print len(sp_list)
    sp_list_set = sorted(list(set(sp_list)))
    print len(sp_list_set)