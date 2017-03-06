import numpy as np
from metric_learn import LMNN , LSML_Supervised
from sklearn.datasets import load_iris
import json


relevant_path = "../sets_jsons/"
#relevant_path = "../sets_jsons_cam/"
included_extenstions = ['json']
file_names = sorted([fn for fn in os.listdir(relevant_path)
              if any(fn.endswith(ext) for ext in included_extenstions)])

   

fn0 = relevant_path + file_names[0]

with open(fn0) as f:
    data = json.load(f)