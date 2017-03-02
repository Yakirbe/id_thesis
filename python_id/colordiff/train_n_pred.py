from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.constraints import *
from keras.regularizers import *
from keras.optimizers import *
import numpy as np
from keras.models import load_model
from scipy import stats


def find_non_zero(W):
    for i in range(len(W)):
        if np.abs(W[i]) >0.000000001:
            print i , W[i]
       

# In keras/metrics.py
def  dice(y_true, y_pred):
    # Symbolically compute the intersection
    y_int = y_true*y_pred
    # Technically this is the negative of the Sorensen-Dice index. This is done for
    # minimization purposes
    return -(2*K.sum(y_int) / (K.sum(y_true) + K.sum(y_pred)))

def id_loss(y_true, y_pred):
    orig = 
    
    

def baseline_model(lendim):

    # create model
    model = Sequential()
    model.add(Dense(lendim, input_dim=lendim, init='normal', W_constraint = nonneg(), activation='linear')) # W_regularizer=l1l2(l1=0.01, l2=0.01),
    model.add(Dense(1, init='normal' , W_constraint = nonneg() , activation='linear'))
    # Compile model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=id_loss, optimizer=sgd)
    return model


    

def trainmodel(lendim ,X_train, y_train , modelname):
    import time
    model = baseline_model(lendim)
    
    #train
    model.fit(X_train, y_train, nb_epoch=3000, batch_size=888)
    time.sleep(0.1)
    #save model
    model.save(modelname)
    
    return model
 



def STRESS(y_pred , Y_te):
    F = (np.sum([yi**2 for yi in y_pred]))/(np.sum([yp*yt for yp,yt in zip(y_pred,Y_te)]))
    top = np.sum([(yp-F*yt)**2 for yp,yt in zip(y_pred,Y_te)])
    down = np.sum([(F**2)*(yt**2) for yt in Y_te])
    return np.sqrt(100*top/down)
    
 
def train_id(X_tr , Y_tr):
    
    X_train = X_tr
    y_train = Y_tr
    
    lendim = len(X_train[0])
    modelname = "linreg.hdf5"
    #score = model.evaluate(X_test, y_test, batch_size=223)
    model = trainmodel(lendim , X_train, y_train , modelname)
    
    return model


def pred_id(model , X_te , Y_te):
    #model = load_model(modelname)
    #load model

    X_test = X_te
    y_test = Y_te
    test_el = np.asarray(X_test[0])
    lendim = len(X_test[0])    
    avg_dis = 0.0
    min_pred = 1000
    max_pred = -1000
    
    y_pred = []
    for i in range(len(Y_te)):
        testel = np.reshape(X_test[i] , (1,lendim))
        #find_non_zero(list(testel[0]))
        y_pred.append(model.predict(testel)[0][0])
        avg_dis += np.abs(model.predict(testel)[0][0]- y_test[i])
        
    print "STRESS = " , STRESS(y_pred , y_test)
        
    avg_dis = avg_dis/len(X_te)
    print "MAE = ", avg_dis
    
    m = np.mean(y_test)
    print stats.ttest_1samp(y_pred, m)
 
if __name__ == "__main__":
    X_train = X_tr
