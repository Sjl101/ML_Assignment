import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans

data = pd.read_csv('Task 2/dog_breeds.csv')
data.head()

def initialise_centroids(dataset, k):
    dt = dataset.to_numpy()
    np.random.shuffle(dt)
    centroids = dt[:k, :]
    return centroids

def compute_euclidean_distance(vec_1, vec_2):
    t = 0 
    for i, value in enumerate(vec_1): t += (vec_2[i] - value) ** 2
    distance = np.sqrt(t)
    return distance

def kmeans(dataset, k):
    data_m = dataset.to_numpy()
    cluster_assigned = np.zeros((len(data_m), 2))
    centroids = initialise_centroids(dataset, k)
    for vi, v in enumerate(data_m):
        centdis = np.zeros(len(centroids))
        for i, centroid in enumerate(centroids):
            d = compute_euclidean_distance(v, centroid)
            centdis[i] = d 
        md = np.min(centdis)
        assigned_cluster = np.argmin(centdis)
        cluster_assigned[vi] = np.array([md, assigned_cluster])  
    cadf = dataset
    cadf['assigned_centroid'] = cluster_assigned[:, 1]
    nc = np.zeros([k, len(centroids[0])])
    for i, centroid in enumerate(centroids):
        cc = cadf
        cc = cc[cc['assigned_centroid'] == i]
        cc = cc.drop(['assigned_centroid'], axis=1)
        current_group_np = cc.to_numpy()
        for x, val in enumerate(centroid):            
            current_column = current_group_np[:, x]
            mean = np.mean(current_column)
            nc[i, x] = mean
        
        
    return centroids, nc, cadf

centroids, mean_cent, clusassdf = kmeans(data, k=3)

km1 = clusassdf[clusassdf['assigned_centroid'] == 0]
km2 = clusassdf[clusassdf['assigned_centroid'] == 1]
km3 = clusassdf[clusassdf['assigned_centroid'] == 2]
#plots the data for the tail length
plt.scatter(km1['height'],km1['tail length'],color='dodgerblue')
plt.scatter(km2['height'],km2['tail length'],color='orange')
plt.scatter(km3['height'],km3['tail length'],color='purple')
plt.scatter(centroids[0, 0],centroids[0, 1],color='black')
plt.scatter(centroids[1, 0],centroids[1, 1],color='black')
plt.scatter(mean_cent[0, 0],mean_cent[1, 1],color='gold')
plt.scatter(mean_cent[1, 0],mean_cent[1, 1,],color='gold')
plt.xlabel('Tail Length')
plt.ylabel('Height')
plt.show()
#plots the data for the leg length
plt.scatter(km1['height'],km1['leg length'],color='cyan')
plt.scatter(km2['height'],km2['leg length'],color='lime')
plt.scatter(km3['height'],km3['leg length'],color='magenta')
plt.scatter(centroids[0, 0],centroids[0, 2],color='black')
plt.scatter(centroids[1, 0],centroids[1, 2],color='black')
plt.scatter(mean_cent[0, 0],mean_cent[1, 2],color='gold')
plt.scatter(mean_cent[1, 0],mean_cent[1, 2],color='gold')
plt.xlabel('Leg Length')
plt.ylabel('Height')
plt.show()