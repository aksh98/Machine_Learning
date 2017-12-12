from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import cPickle, gzip, numpy
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib
import h5py
from random import shuffle

    
def load_h5py(filename):
    with h5py.File(filename, 'r') as hf:
        X = hf['X'][:]
        Y = hf['Y'][:]
    return X, Y


[X,Y] = load_h5py('dataset_partA.h5')

X = np.array(X)
Y = np.array(Y)

X = np.reshape(X,(len(X),len(X[0])*len(X[0])))
X_train,Y_train, Test_x, test_y = sklearn.model_selection.train_test_split(X,Y,test_size = 0.20)

#print(len(training_data),len(training_data[0]),len(training_data[0][0]), len(test_data),len(test_data[0]))

training_Data = []
input_data = []
val_data = []
input_data = []
t_data = []
test_inp_X = []
val_input_X = []
tr_input_X = []
tr_output_Y = []
size_inputLayer = 784 
# for i in training_Data:
# 	tr_input_X.append(np.reshape(i,(size_inputLayer,1)))

# for i in val_data[0]:
# 	val_input_X.append(np.reshape(i,(size_inputLayer,1)))
# for i in test_data[0]:
# 	test_inp_X.append(np.reshape(i,(size_inputLayer,1)))

mlp = MLPClassifier(activation='logistic',hidden_layer_sizes=(100,50), max_iter=100, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X_train, Y_train)
print("Training set score: ",mlp.score(X_train, Y_train))
print("Test set score: ",mlp.score(Test_x,test_y))

# ('Training set score: ', 0.99748000000000003)
# ('Test set score: ', 0.97550000000000003)