from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import cPickle, gzip, numpy
from sklearn.datasets import fetch_mldata

f = gzip.open('mnist.pkl.gz', 'rb')
training_data, validation_data, test_data = cPickle.load(f)
f.close()
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

mlp = MLPClassifier(activation='relu',hidden_layer_sizes=(100,50), max_iter=100, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4,
                    learning_rate_init=0.1)

mlp.fit(training_data[0], training_data[1])
print("Training set score: ",mlp.score(training_data[0], training_data[1]))
print("Test score: ",mlp.score(test_data[0],test_data[1]))

# ('Training set score: ', 0.99748000000000003)
# ('Test set score: ', 0.97550000000000003)