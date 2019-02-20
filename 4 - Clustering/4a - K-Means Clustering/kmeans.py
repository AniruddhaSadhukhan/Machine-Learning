# K-Means Clustering 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, -2:].values

# Using the elbow method to find optimal number of clusters
from sklearn.cluster import KMeans
WCSS = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,
                    init = 'k-means++',
                    max_iter = 300,
                    n_init = 10,
                    random_state = 0)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)

plt.plot(range(1,11),WCSS)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying K-means to the dataset
kmeans = KMeans(n_clusters = 5,
                    init = 'k-means++',
                    max_iter = 300,
                    n_init = 10,
                    random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Visualising the results
plt.scatter(X[y_kmeans == 0, 0],X[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Careful')
plt.scatter(X[y_kmeans == 1, 0],X[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Standard')
plt.scatter(X[y_kmeans == 2, 0],X[y_kmeans == 2, 1], s = 50, c = 'yellow', label = 'Target')
plt.scatter(X[y_kmeans == 3, 0],X[y_kmeans == 3, 1], s = 50, c = 'magenta', label = 'Careless')
plt.scatter(X[y_kmeans == 4, 0],X[y_kmeans == 4, 1], s = 50, c = 'cyan', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], marker='*', s = 300, c = 'black', label = 'Centroids')
plt.title('Cluster of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend() 
plt.show()

