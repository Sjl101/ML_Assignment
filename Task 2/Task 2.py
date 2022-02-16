import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans

data = pd.read_csv('Task 2/dog_breeds.csv')
data.head()

def compute_euclidean_distance(vec_1, vec_2):
    t = 0 
    for i, value in enumerate(vec_1): t += (vec_2[i] - value) ** 2
    distance = np.sqrt(t)
    return distance

def initialise_centroids(dataset, k):
    dt = dataset.to_numpy()
    np.random.shuffle(dt)
    centroids = dt[:k, :]
    return centroids

def kmeans(dataset, k):
    data_m = dataset.to_numpy()
    cluster_assigned = np.zeros((len(data_m), 2))
    centroids = initialise_centroids(dataset, k)
    for v_index, vector in enumerate(data_m):
        centroid_distances = np.zeros(len(centroids))
        for centroid_i, centroid in enumerate(centroids):
            distance = compute_euclidean_distance(vector, centroid)
            centroid_distances[centroid_i] = distance  
        min_distance = np.min(centroid_distances)
        assigned_cluster = np.argmin(centroid_distances)
        cluster_assigned[v_index] = np.array([min_distance, assigned_cluster])  
    cluster_assigned_df = dataset.copy()
    cluster_assigned_df['assigned_centroid'] = cluster_assigned[:, 1]
    new_centroids = np.zeros([k, len(centroids[0])])
    for centroid_i, centroid in enumerate(centroids):
        current_centroid = cluster_assigned_df.copy()
        current_centroid = current_centroid[current_centroid['assigned_centroid'] == centroid_i]
        current_centroid = current_centroid.drop(['assigned_centroid'], axis=1)
        current_group_np = current_centroid.values
        for x, val in enumerate(centroid):            
            current_column = current_group_np[:, x]
            mean = np.mean(current_column)
            new_centroids[centroid_i, x] = mean
        
        
    return centroids, new_centroids, cluster_assigned_df

centroids, mean_cent, clusassdf = kmeans(data, k=3)
print('The centroids: ', centroids)
print('The mean centroids: ', mean_cent)

km1 = clusassdf[clusassdf['assigned_centroid'] == 0]
km2 = clusassdf[clusassdf['assigned_centroid'] == 1]
km3 = clusassdf[clusassdf['assigned_centroid'] == 2]
plt.scatter(km1['height'],km1['tail length'],color='blue')
plt.scatter(km2['height'],km2['tail length'],color='red')
plt.scatter(km3['height'],km3['tail length'],color='purple')
plt.scatter(centroids[0, 0],centroids[0, 1],color='black')
plt.scatter(centroids[1, 0],centroids[1, 1],color='black')
plt.scatter(mean_cent[0, 0],mean_cent[1, 1],color='yellow')
plt.scatter(mean_cent[1, 0],mean_cent[1, 1,],color='yellow')
plt.show()
plt.scatter(km1['height'],km1['leg length'],color='cyan')
plt.scatter(km2['height'],km2['leg length'],color='lime')
plt.scatter(km3['height'],km3['leg length'],color='magenta')
plt.scatter(centroids[0, 0],centroids[0, 2],color='black')
plt.scatter(centroids[1, 0],centroids[1, 2],color='black')
plt.scatter(mean_cent[0, 0],mean_cent[1, 2],color='yellow')
plt.scatter(mean_cent[1, 0],mean_cent[1, 2],color='yellow')
plt.show()