import random
import pandas
import csv
import numpy
import matplotlib 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
# from tsne import bh_sne
# import sklearn
from sklearn.metrics import cluster

def k_means(k,k_cent, num_points, array, dim):
    #k centroids
    c_arr = []
    dist = 0
    min_dist = 0
    #nearest centroid for each point
    cent = []
    for i in range(0,k_cent):
        c_arr.append(array[random.randint(0,num_points)])
    # print(c_arr)
    obj_func = []
    for x in range(0,20):
        #find nearest centroid
        for i in range(0,num_points):
            min_dist = 999999
            if(x==0):
                cent.append(0)
            for j in range(0,k_cent):
                dist = numpy.linalg.norm(array[i]-c_arr[j])
                if min_dist > dist:
                    min_dist = dist
                    cent[i] = j
        #update centroids
        num = 0
        total = 0
        lala = k_cent
        while(lala > 0):
            lala = lala-1
            mysum = numpy.zeros(dim+1)
            num = 0
            for i in range(0,num_points):
                if(cent[i] == lala):
                    num = num+1
                    mysum = mysum+array[i]                    
            #new centroids
            c_arr[lala] = mysum/num
            total = total + mysum
        print(c_arr)
        obj_func.append(sum(total))
    print("----------MY ARRAY-------------------------------------")
    print (c_arr)
    return cent,c_arr,obj_func
    

k = 3
dataset = pandas.read_csv('seedset.csv')
print(dataset.shape)
# print dataset
data = numpy.array(dataset)
# print data
or_labels = []
for i in range(0,dataset.shape[0]):
    or_labels.append(data[i][dataset.shape[1]-1])
    data[i][dataset.shape[1]-1] = 0
# print data
mylabels = []
k_cent = []
for i in or_labels:
    if i not in k_cent:
        k_cent.append(i)
# print(k_cent)
mylabels,c_aerr,obj_func = k_means(k_cent,k,dataset.shape[0], data, dataset.shape[1]-1)
# print(len(obj_func))
# #####---SKLEARN-----#####
# sklearn_kmeans = KMeans(n_clusters=3)
# sklearn_kmeans = sklearn_kmeans.fit(data)
# labels = sklearn_kmeans.predict(data)
# centroids = sklearn_kmeans.cluster_centers_
# print(centroids)
######--------##########
print(cluster.adjusted_rand_score(or_labels, mylabels))
print(cluster.normalized_mutual_info_score(or_labels, mylabels))
print(cluster.adjusted_mutual_info_score(or_labels, mylabels))
# pyplot.plot(data)

###########################
# tsne = TSNE(n_components=2, random_state=0)
# X_tsne = tsne.fit_transform(data)
# ce = tsne.fit_transform(c_aerr)
# plt.figure()
# plt.scatter(X_tsne[:,0],X_tsne[:,1], c=or_labels,s = 20 , cmap='viridis')
# plt.scatter(ce[:, 0], ce[:, 1], c='black', s=80, alpha=0.6);
##############################
#plot obj func graph
arr = []
for i in range(0,20):
    arr.append(i)
# print(len(arr))
print(obj_func)
print(arr)
plt.plot(arr,obj_func)

plt.show()