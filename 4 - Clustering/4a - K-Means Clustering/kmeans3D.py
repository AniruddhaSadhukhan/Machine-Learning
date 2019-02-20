# K-Means Clustering 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, -3:].values

# Using the elbow method to find optimal number of clusters
from sklearn.cluster import KMeans
WCSS = []
for i in range(1,21):
    kmeans = KMeans(n_clusters = i,
                    init = 'k-means++',
                    max_iter = 300,
                    n_init = 10,
                    random_state = 0)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)

plt.figure(1)
plt.plot(range(1,21),WCSS)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying K-means to the dataset
kmeans = KMeans(n_clusters = 6,
                    init = 'k-means++',
                    max_iter = 300,
                    n_init = 10,
                    random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Visualising the results
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[y_kmeans == 0, 0],X[y_kmeans == 0, 1],X[y_kmeans == 0, 2], s = 50, c = 'red', label = 'Careful')
ax.scatter(X[y_kmeans == 1, 0],X[y_kmeans == 1, 1],X[y_kmeans == 1, 2], s = 50, c = 'blue', label = 'Standard')
ax.scatter(X[y_kmeans == 2, 0],X[y_kmeans == 2, 1],X[y_kmeans == 2, 2], s = 50, c = 'yellow', label = 'Target')
ax.scatter(X[y_kmeans == 3, 0],X[y_kmeans == 3, 1],X[y_kmeans == 3, 2], s = 50, c = 'magenta', label = 'Careless')
ax.scatter(X[y_kmeans == 4, 0],X[y_kmeans == 4, 1],X[y_kmeans == 4, 2], s = 50, c = 'cyan', label = 'Sensible')
ax.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,2], marker='*', s = 300, c = 'black', label = 'Centroids')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending score (1-100)')
plt.title('Cluster of Clients')
plt.legend()
plt.show()


