import sklearn
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib
import h5py
import os
import os.path
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
import argparse
from sklearn.manifold import TSNE
from sklearn.multiclass import OneVsRestClassifier
from sklearn import datasets,svm
from sklearn.metrics import confusion_matrix
from scipy import interp

def OneVsRestClassif():
	return arr


def remove_outliers(X,y):
	mean = np.mean(X, axis=0)
	std = np.std(X, axis=0)
	final_list = np.array()
	for x in arr:
		if (x>(mean - 2*std)) and (x<(mean + 2*std)):
			final_list.append(x)
	return final_list


def prediict(kernel,test,intercpt,dualcoef,suppvec):
	temp = np.zeros(len(test))
	#print(len(test))
	#print(intercpt)
	for j in range(len(test)):
		temp[j] = intercpt
		for i in range(0,len(suppvec)):
			if(kernel =='linear'):
				temp[j] = temp[j] + dualcoef[0,i]*np.dot(suppvec[i],test[j])
				print('ola')
			#print(temp[j])
			elif(kernel == 'rbf'):
				temp[j] = temp[j] + dualcoef[0,i]*np.exp((-9*np.linalg.norm(suppvec[i]-test[j])**2))
				#print('olaa')
			else:
				break
		#if(kernel == 'linear'):
		if(temp[j] < 0):
			temp[j] = 0
		else:
			temp[j] = 1
	#print(temp[j])
	return temp	

#def kernel(X,Y):
#	for i in X.shape():
#		x[i] = abs(x[i])
#		y[i] = abs(y[i])

def newk(x,y):
	gamma = 0.1
	return np.exp((-1*np.linalg.norm(x-y)**2))
#rbf - gaussian kernel !
def kernel(X,Y):
    Krnl = np.zeros((X.shape[0],Y.shape[0]))
    for i,x in enumerate(X):
        for j,y in enumerate(Y):
        	#kernel update
            Krnl[i,j] = newk(x,y)
    return Krnl

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )
args = parser.parse_args()

filename = args.data

hf = h5py.File(filename,'r')
features = hf.keys()[0]
clss = hf.keys()[1]

x = np.array(list(hf[features]))
arr = np.array(list(hf[clss])) 
print(x.shape)
print(arr.shape)
y = np.array(len(arr))
for i in range(0,len(arr)):
	for j in range(0,len(arr[i])):
		if(arr[i][j] == 1):
			y[i] = j

y = y[0:len(y)-200]
x = y[0:len(y)-200]
# remove_outliers(x,y)

k = 5
a = 0
x_test = x[a:a+(len(y)/k),:]
y_test = y[a:a+(len(y)/k)]
y_train = y[a+(len(y)/k):len(y)]
x_train = np.concatenate((x[:a,],x[a+(len(y)/k):,]),axis = 0)

# clf=SVR(kernel=my_kernel)
#decision shape - ovo/ovr ! 
# model = OneVsRestClassifier(svm.SVC(kernel=kernel))#SVC(kernel=kernel) 

model = svm.SVC(kernel='rbf')

model.fit(x_train,y_train)

dualcoef = model.dual_coef_
intercpt = model.intercept_
suppvec = model.support_vectors_
kern = model.kernel
xmin,xmax = min(x[:,0])-1,max(x[:,0])+1
ymin,ymax = min(x[:,1])-1,max(x[:,1])+1
xx,yy = np.meshgrid(np.arange(xmin,xmax,0.01),np.arange(ymin,ymax,0.01))
	#self implemented
Z = prediict('rbf',x_test,0,dualcoef,suppvec)
# print(y_test.shape)
# print(Z.shape)
# print(x_test.shape)
# print(y_train.shape)
# print("lalal")
#print(model.score(y_test,Z))
#print(accuracy_score(y_test,Z))
y_score =  accuracy_score(y_test,Z)
print "accuracy_score of ", filename ,"is : ", y_score
	#original

# confusion = confusion_matrix(y_test, Z)
# confusion = confusion.astype('float')/confusion.sum(axis=1)[:, np.newaxis]
# print("Confusion matrix with Normalization")
# print(confusion)

###############################################################
# Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# out = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
# plt.scatter(x[:,0],x[:,1], c=y, cmap=plt.cm.coolwarm, s=30, edgecolors='k')
# plt.title('RBF Kernel - Plot 3')
# plt.xlabel('Parame')
# plt.ylabel('Accuracy')
# plt.show()

