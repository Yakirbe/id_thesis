import numpy as np
from metric_learn import LMNN , LSML_Supervised
from sklearn.datasets import load_iris
#
#iris_data = load_iris()
#X_iris = iris_data['data']
#Y_iris = iris_data['target']
#
#lmnn = LMNN(k=5, learn_rate=1e-6)
#lmnn.fit(X, Y)#, verbose=False)




lsml = LSML_Supervised(num_constraints=200)
lsml.fit(X, Y)