import os
import os.path
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )
args = parser.parse_args()


def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y


[X,Y] = load_h5py(args.data)
filename = args.data
X_Output = TSNE(n_components =2).fit_transform(X) #n_components = 2

arr = []
for i in range(0,len(Y)):
	for j in range(0,len(Y[i])):
		if(Y[i][j]==1):
			arr.append(j)

			
plt.figure(figsize=(12, 8))
plt.scatter(X_Output[:,0], X_Output[:,1], s=30, c=arr, cmap=plt.cm.Paired)
plt.show()
plt.clf()