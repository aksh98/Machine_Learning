import sklearn
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib
import h5py
import os
import os.path
import argparse
from sklearn.manifold import TSNE
from sklearn import datasets,svm

def newk(x,y):
	gamma = 0.1
	return np.exp((-9*np.linalg.norm(x-y)**2))
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
y = np.array(list(hf[clss])) 
# model = svm.SVC(kernel='rbf', C=5,degree=3) 
model = svm.SVC(kernel=kernel) 

#1-rbf
#2-rbf 
#3-linear 
#4-rbf
#5-rbf
model.fit(x,y)

xmin,xmax = min(x[:,0])-1,max(x[:,0])+1
ymin,ymax = min(x[:,1])-1,max(x[:,1])+1
xx,yy = np.meshgrid(np.arange(xmin,xmax,0.01),np.arange(ymin,ymax,0.01))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
out = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
 
plt.scatter(x[:,0],x[:,1], c=y, cmap=plt.cm.coolwarm, s=30, edgecolors='k')
plt.show()

# plt.figure(figsize=(12, 8))
# plt.scatter(x[:,0], x[:,1], s=30, c=y, cmap=plt.cm.Paired)
# plt.show()
# plt.clf()

