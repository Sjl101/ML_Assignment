import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans

data = pd.read_csv('Task 2/dog_breeds.csv')
data.head()

def initialise_centroids(dataset, k):
    dt = dataset.to_numpy()
    #shuffels the data
    np.random.shuffle(dt)
    #gets k number of row
    centroids = dt[:k, :]
    return centroids

def compute_euclidean_distance(vec_1, vec_2):
    t = 0 
    for i, value in enumerate(vec_1): t += (vec_2[i] - value) ** 2
    distance = np.sqrt(t)
    return distance

def kmeans(dataset, k):
    dm = dataset.to_numpy()
    #stores assigned cluster and distance
    ca = np.zeros((len(dm), 2))
    centroids = initialise_centroids(dataset, k)
    #measures the distance of each point in each row in the dataset
    for vi, v in enumerate(dm):
        centdis = np.zeros(len(centroids))
        for i, centroid in enumerate(centroids):
            d = compute_euclidean_distance(v, centroid)
            centdis[i] = d 
        md = np.min(centdis)
        ac = np.argmin(centdis)
        ca[vi] = np.array([md, ac])
    #creats a new dataset with the chosen clusters as the new column   
    cadf = dataset
    cadf['ac'] = ca[:, 1]
    nc = np.zeros([k, len(centroids[0])])
    #calculates the mean of the clusters
    for i, centroid in enumerate(centroids):
        cc = cadf
        cc = cc[cc['ac'] == i]
        cc = cc.drop(['ac'], axis=1)
        cgnp = cc.to_numpy()
        #updates the centroid with mean of each cluster
        for x, val in enumerate(centroid):            
            cc = cgnp[:, x]
            mean = np.mean(cc)
            nc[i, x] = mean   
    return centroids, nc, cadf
centroids, mean_cent, clusassdf = kmeans(data, k=3)

km1 = clusassdf[clusassdf['ac'] == 0]
km2 = clusassdf[clusassdf['ac'] == 1]
km3 = clusassdf[clusassdf['ac'] == 2]

#plots the data for the tail length
plt.scatter(km1['height'],km1['tail length'],color='dodgerblue')
plt.scatter(km2['height'],km2['tail length'],color='orange')
plt.scatter(km3['height'],km3['tail length'],color='purple')
plt.scatter(centroids[0, 0],centroids[0, 1],color='black')
plt.scatter(centroids[1, 0],centroids[1, 1],color='black')
plt.scatter(mean_cent[0, 0],mean_cent[1, 1],color='gold')
plt.scatter(mean_cent[1, 0],mean_cent[1, 1,],color='gold')
plt.xlabel('Height')
plt.ylabel('Tail Length')
plt.savefig('tail_length.png')
plt.show()

#plots the data for the leg length
plt.scatter(km1['height'],km1['leg length'],color='cyan')
plt.scatter(km2['height'],km2['leg length'],color='lime')
plt.scatter(km3['height'],km3['leg length'],color='magenta')
plt.scatter(centroids[0, 0],centroids[0, 2],color='black')
plt.scatter(centroids[1, 0],centroids[1, 2],color='black')
plt.scatter(mean_cent[0, 0],mean_cent[1, 2],color='gold')
plt.scatter(mean_cent[1, 0],mean_cent[1, 2],color='gold')
plt.xlabel('Height')
plt.ylabel('Leg Length')
plt.savefig('leg_length.png')
plt.show()