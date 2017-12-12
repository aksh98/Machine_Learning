import sklearn
import matplotlib.pyplot as plt 
import numpy as np
from numpy import savetxt
import matplotlib
import h5py
import os
import os.path
import argparse
from sklearn.manifold import TSNE
from sklearn import datasets,svm
import json
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )
args = parser.parse_args()

filename = args.data
#file2 = args.test
data = []
test = []

with open('test.json','r') as ff:
 	test = json.load(ff)

with open(filename) as f:
    data = json.load(f)


arr1 = []
arr2 = []
ezee = []
arr = []
for i in range(len(data)):
	arr1.append(data[i]['X'])
	arr2.append(data[i]['Y'])
	
for i in range(len(test)):	
	arr.append(test[i]['X'])

# x.shape
x = np.array(arr1)
y = np.array(arr2)
testt = np.array(arr)

for i in x:
	strin = ' '.join(str(ee) for ee in i)
	ezee.append(strin)
ezee = np.array(ezee)	

diff = []
for i in testt:
	strin = ' '.join(str(ee) for ee in i)
	diff.append(strin)
diff = np.array(diff)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train = vectorizer.fit_transform(ezee)
X_test = vectorizer.transform(diff)


classifier = LinearSVC(penalty='l2',C=0.1, loss='squared_hinge',max_iter=1000, dual=True,tol=0.00002)
classifier.fit(X_train,y)
b = classifier.predict(X_test)
a = np.arange(1,len(b)+1)

fi = open('sub6.csv', 'wb')
fi.write("{},{}\n".format("Id", "Expected"))
for xa in zip(a, b):
    fi.write("{},{}\n".format(xa[0], xa[1]))

fi.close()
