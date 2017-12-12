import random
import gzip
# from mnist import MNIST
import numpy as np
import cPickle
import pickle
from random import shuffle

import random
import gzip
# from mnist import MNIST
import numpy as np
import cPickle
import pickle
from random import shuffle


def relu(z):
    # for ReLU use this line
    return 1.0 * (z > 0)

def drelu(z):
  #for reLU use this
    t = 1.0 * (z > 0)
    t[t == 0] = 0.1
    return t

def ff(wt,bias,activation):
    for bi, weit in zip(bias, wt):
        activation = relu(np.dot(weit, activation)+bi)
        #print
    return activation

def backprop( num_layers,wt,bias,num_act, y):
    tuple_bias = [np.zeros(b.shape) for b in bias]
    tuple_w = [np.zeros(w.shape) for w in wt]
    #print(tuple_w)           
    # ff
    activate = num_act
    stimule = [num_act]
    z_vectors = []
    for biass, w in zip(bias,wt):
       # print(w,activate)
        temp = np.dot(w, activate)+biass
        z_vectors.append(temp)
        activate = relu(temp)
        stimule.append(activate)
    # backward pass
#         print(len(stimule[-1]),len(y))
#         exit(1)
    tiny_change = (stimule[-1] - y) * drelu(z_vectors[-1])
    tuple_w[-1] = np.dot(tiny_change, stimule[-2].transpose())
    #print("lals")
    tuple_bias[-1] = tiny_change
    for layer in range(2, num_layers):
        temp = z_vectors[-layer]
        sp = drelu(temp)
        tiny_change = np.dot(wt[-layer+1].transpose(), tiny_change) * sp
        tuple_w[-layer] = np.dot(tiny_change, stimule[-layer-1].transpose())
        tuple_bias[-layer] = tiny_change
    return (tuple_bias, tuple_w)

def Stoc_Grad_Desc(training, epochs, mini_batch_size, learning_rate,
        test):
    NN_Len = [784,100,50,10]
    num_layers = len(NN_Len)
    # print(NN_Len[1:])
    wt = []
    bias = []
    for i, j in zip(NN_Len[:-1], NN_Len[1:]):
        wt.append(np.random.randn(j, i))
    for i in NN_Len[1:]:
        bias.append(np.random.randn(i, 1))
    
    slots = []
    tuple_bias = []
    tuple_w = []
    for j in range(0,epochs):
        shuffle(training)
        for k in range(0, len(training), mini_batch_size):
            slots.append(training[k:k+mini_batch_size])
        for split_data in slots:
            #update_mini_batch(split_data, learning_rate)
            tuple_bias = [np.zeros(b.shape) for b in bias]
            tuple_w = [np.zeros(w.shape) for w in wt]
            
            for x, y in split_data:
                change_b, change_wt = backprop(num_layers,wt,bias,x, y)

                tuple_w = [dw+num_w for num_w, dw in zip(tuple_w, change_wt)]

                tuple_bias = [num_b+dnb for num_b, dnb in zip(tuple_bias, change_b)]
            wt = [w-(learning_rate/len(split_data))*num_w for w, num_w in zip(wt, tuple_w)]
            bias = [b-(learning_rate/len(split_data))*num_b for b, num_b in zip(bias, tuple_bias)]
        test_results = []
        #print(len(test))
        for (x, y) in test:
            test_results.append((np.argmax(ff(wt,bias,x)), y))
        #lala = sum(int(x == y) for (x, y) in test_results)
        loco = 0
        for x,y in test_results:
            if x-y<2:
                loco = loco + 1
        accuracy = (float(loco)/len(test))*100
        #print(len(wt))
        print ("Epoch :" ,j, "Accuracy : ", accuracy)

    

#### ---------MAIN-------------------------------------
def main():
    



    #//////////////////////////////////////////

    file = gzip.open('mnist.pkl.gz', 'rb')
    tr_data, validation_data, t_data = cPickle.load(file)
    f.close()
    print("Data loaded !")
    tr_inp_X = []
    tr_output_Y = []
    # validation_inputs = []
    test_inputs = []
    
    for i in tr_data[0]:
        tr_inp_X.append(np.reshape(i, (784, 1)))
    #print(tr_inp_X.shape)

    # for i in validation_data[0]:
    #     validation_inputs.append(np.reshape(i, (784, 1)))
    # validation_data = zip(validation_inputs, validation_data[1])
    for x in t_data[0]:
        test_inputs.append(np.reshape(x, (784, 1)))
    t_data = zip(test_inputs, t_data[1])    

    for y in tr_data[1]:
        s = np.zeros((10,1))
        s[y] = 1.0
        tr_output_Y.append(s)
    training = zip(tr_inp_X, tr_output_Y)

    print(len(training),len(training[0][0]),len(training[0]))
    #training = training + validation_data
    #print(len(training),len(training[0][0]),len(training[0]))

    Stoc_Grad_Desc(training, 20, 10, 0.3 ,t_data)

main()

