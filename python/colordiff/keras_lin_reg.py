from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.constraints import *
from keras.regularizers import *
from keras.optimizers import *
import numpy as np
from keras.models import load_model



def find_non_zero(W):
    for i in range(len(W)):
        if np.abs(W[i]) >0.000000001:
            print i , W[i]
            
            
            
            
def baseline_model(lendim):

    # create model
    model = Sequential()
    model.add(Dense(lendim, input_dim=lendim, init='normal', W_constraint = nonneg(), activation='linear')) # W_regularizer=l1l2(l1=0.01, l2=0.01),
    model.add(Dense(1, init='normal' , W_constraint = nonneg()))
    # Compile model
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    adg = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=adg)
    return model



def trainmodel(lendim ,X_train, y_train ):
    model = baseline_model(lendim)
    
    #train
    model.fit(X_train, y_train, nb_epoch=20000, batch_size=888)
    time.sleep(0.1)
    #save model
    model.save("linreg.hdf5")
    
    return model
 
X_train = X_tr
y_train = Y_tr

X_test = X_te
y_test = Y_te

lendim = len(X_test[0])

#score = model.evaluate(X_test, y_test, batch_size=223)
model = trainmodel(lendim , X_train, y_train)
#model = load_model("linreg.hdf5")
#load model
test_el = np.asarray(X_test[0])



avg_dis = 0.0
for i in range(len(Y_te)):
    print i
    testel = np.reshape(X_test[i] , (1,lendim))
    find_non_zero(list(testel[0]))
    print y_test[i] ,  model.predict(testel)[0][0]
    print
    avg_dis += np.abs(model.predict(testel)[0][0]- y_test[i])

    
avg_dis = avg_dis/len(X_te)
print "avg dis = ", avg_dis