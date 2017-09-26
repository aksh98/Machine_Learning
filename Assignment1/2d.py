import matplotlib.pyplot as plt 
import numpy as np 
import sklearn 
import h5py
import matplotlib
import argparse
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression

# x = features
# y - labels
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str)
parser.add_argument("--weights_path", type = str)
parser.add_argument("--train_data", type = str)
parser.add_argument("--plots_save_dir", type = str)
args = parser.parse_args()
[X,Y] = load_h5py("part_A_train.h5")
X_Output = TSNE(n_components=2).fit_transform(X)
print("y-output: ",X_Output.shape)
print("y-shape: ",Y.shape)
arr = []
for i in range(0,len(Y)):
	for j in range(0,len(Y[i])):
		if(Y[i][j]==1):
			arr.append(j)

k = 20
a = 0
a1 = 0
a2 = 0
a3 = 0
x_train = []
y_train = []
x_test = []
y_test = []
#for i in range(0,k):

#### TEST_TRAIN_DATA ###########################################	
a = 2*((len(arr))/k)
x_test = X[a:a+(len(arr)/k),:]
y_test = arr[a:a+(len(arr)/k)]
x_train = np.concatenate((X[:a,],X[a+(len(arr)/k):,]),axis = 0)
y_train = arr[0:a]+arr[a+(len(arr)/k):len(arr)]

#### GAUSSIAN-NB ###############################################
gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_predict = gnb.predict(x_test)
a1 = accuracy_score(y_test,y_predict,normalize = True)
#train_score,valid_score = validation_curve(GaussianNB(),)

###### DECISION TREEE ################################################
DT_file = 'decision_tree_classifier.pkl'
DT_model_pkl = open(DT_file,'wb')
diction = {'cr':['gini','entropy'],'depth':[5,7,9,11,13,15],'split':[5,10,15,20]}
acc=0
for i in diction['cr']:
	for j in diction['split']:
		for kn in diction['depth']:
			dtc = DecisionTreeClassifier(criterion = i, random_state = 42,
  	                            max_depth=kn,min_samples_split=j,min_samples_leaf=10)
			dtc.fit(x_train, y_train)
			pickle.dump(dtc,DT_model_pkl)
			t = dtc.predict(x_test)
			if(acc < accuracy_score(y_test,t,normalize=True)):
				acc = accuracy_score(y_test,t,normalize=True)
				cr = i
				split = j
				depth = kn
# a2 = a2+acc
DT_model_pkl.close()
#### Logistic Regression ##########################################################
axx = 0

dict = {'c':[1.0,100.0,1000.0,10000.0],'solv':['newton-cg','lbfgs','sag','saga']}
for i in dict['c']:
	for j in dict['solv']:
		lregg= LogisticRegression(multi_class='multinomial', solver=j
					,C=i,fit_intercept=True,random_state=None)
		lregg.fit(x_train,y_train)
		if(axx < accuracy_score(y_test,lregg.predict(x_test),normalize=True)):
			axx = accuracy_score(y_test,lregg.predict(x_test),normalize=True)
			c = i
			solv = j
# a3 = a3 + axx
#	print "-----------------------------------------\n"
print "\nAccuracy score |  GaussianNB  : ",a1
print "Accuracy score | Decision Tree: ",acc
print "Accuracy score | Logistc Regrn: ",axx



##########################################

# plt.figure(figsize=(12, 8))
# plt.scatter(X_Output[:,0], X_Output[:,1], s=30, c=arr, cmap=plt.cm.Paired)
# plt.show()
# plt.clf()