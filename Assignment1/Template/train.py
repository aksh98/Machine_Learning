import matplotlib.pyplot as plt 
import os
import os.path
import argparse
import h5py
import numpy as np
import sklearn
import pickle
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str  )
parser.add_argument("--weights_path", type = str)
parser.add_argument("--train_data", type = str  )
parser.add_argument("--plots_save_dir", type = str)
args = parser.parse_args()
filename = args.train_data
# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

[X,Y] = load_h5py(filename)
X_Output = TSNE(n_components=2).fit_transform(X)
print("y-output: ",X_Output.shape)
print("y-shape: ",Y.shape)

arr = []
for i in range(0,len(Y)):
	for j in range(0,len(Y[i])):
		if(Y[i][j]==1):
			arr.append(j)

# Preprocess data and split it
a1 = 0
k = 5
a = 0
a2 = 0
a3 = 0
x_train = []
y_train = []
x_test = []
y_test = []
# Train the models

#### GAUSSIAN ###########################################	
if args.model_name == 'GaussianNB':
	# Train the models
	for i in range(0,k):
		a = i*((len(arr))/k)
		x_test = X[a:a+(len(arr)/k),:]
		y_test = arr[a:a+(len(arr)/k)]
		x_train = np.concatenate((X[:a,],X[a+(len(arr)/k):,]),axis = 0)
		y_train = arr[0:a]+arr[a+(len(arr)/k):len(arr)]
		gnb = GaussianNB()
		gnb.fit(x_train,y_train)
		y_predict = gnb.predict(x_test)
		a1 = a1 + accuracy_score(y_test,y_predict,normalize = True)
	print "\nAccuracy score |  GaussianNB  : ",float(a1)/k

#### LOGISTICREGRESSION ###########################################	
elif args.model_name == 'LogisticRegression':

	Lmatrix = [[0 for x in range(16)] for y in range(k)] 
	Lnest = [[0 for x in range(2)] for y in range(16)]
	mazz = 0
	for i in range(0,k):
		a = i*((len(arr))/k)
		# Train the models
		x_test = X[a:a+(len(arr)/k),:]
		y_test = arr[a:a+(len(arr)/k)]
		x_train = np.concatenate((X[:a,],X[a+(len(arr)/k):,]),axis = 0)
		y_train = arr[0:a]+arr[a+(len(arr)/k):len(arr)]
		axx = 0
		count = 0
		dict = {'c':[1.0,10.0,30.0,50.0],'solv':['newton-cg','lbfgs','sag','saga']}
		for x in dict['c']:
			for j in dict['solv']:
				lregg= LogisticRegression(multi_class='multinomial',solver=j,C=x,fit_intercept=True)
				lregg.fit(x_train,y_train)
				temp = accuracy_score(y_test,lregg.predict(x_test),normalize=True)
				Lmatrix[i][count] = temp
				Lnest[count][0] = x
				Lnest[count][1] = j
				count = count+1
				if(axx < temp):
					axx = temp
					mazz = lregg
		a3 = a3 + axx
	LR_file='LRpkl.pkl'
	LRmodelpkl = open(LR_file,'wb')
	pickle.dump(mazz,LRmodelpkl)
	LRmodelpkl.close()
	Lmat = np.mean(Lmatrix,axis = 0)
	print "Accuracy score | Logistc Regrn: ",float(a3)/k
	
	plt.bar(range(len(Lnest)),Lmat)
	plt.title('LogisticRegr Plot')
	plt.xlabel('Parameters')
	plt.ylabel('Accuracy')
	plt.show()
#### DECISION TREE #####################################################
elif args.model_name == 'DecisionTreeClassifier':
	# define the grid here
	diction = {'depth':[5,8,11,15],'split':[5,10,15,20]}
	# do the grid search with k fold cross validation
	Dmatrix = [[0 for x in range(16)] for y in range(k)] #Dmatrix[k+1][16]
	Dnest =[[0 for x in range(2)] for y in range(16)]
	mazza = 0
	for i in range(0,k):
		# Train the models
		a = i*((len(arr))/k)
		x_test = X[a:a+(len(arr)/k),:]
		y_test = arr[a:a+(len(arr)/k)]
		x_train = np.concatenate((X[:a,],X[a+(len(arr)/k):,]),axis = 0)
		y_train = arr[0:a]+arr[a+(len(arr)/k):len(arr)]
		acc=0
		count=0
		for j in diction['split']:
			for kn in diction['depth']:
				# model = DecisionTreeClassifier(  ...  )
				dtc = DecisionTreeClassifier(criterion = 'gini', random_state = 42,
	  	                            max_depth=kn, min_samples_split=j, min_samples_leaf=10)
				dtc.fit(x_train, y_train)
				t = dtc.predict(x_test)
				temp = accuracy_score(y_test,t,normalize=True)
				Dmatrix[i][count] = temp
				Dnest[count][0] = j
				Dnest[count][1] = kn 
				count = count + 1
				if(acc < temp):
					acc = temp
					mazza = dtc
		a2 = a2+acc
	Dmat = np.mean(Dmatrix,axis = 0)
	DT_file='DTpkl.pkl'
	DTmodelpkl = open(DT_file,'wb')
	pickle.dump(mazza,DTmodelpkl)
	DTmodelpkl.close()
	print "Accuracy score | Decision Tree: ",float(a2)/k
	plt.bar(range(len(Dnest)),Dmat)
	plt.title('LogisticRegr Plot')
	plt.xlabel('Parameters')
	plt.ylabel('Accuracy')
	plt.show()
	# save the best model and print the results

else:
	raise Exception("Invald Model name")