# Hierarchical Clustering 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, -2:].values

# Using the Dendogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5,
                             affinity = 'euclidean',
                             linkage = 'ward')
y_hc = hc.fit_predict(X)


# Visualising the results
plt.figure(2)
plt.scatter(X[y_hc == 0, 0],X[y_hc == 0, 1], s = 50, c = 'red', label = 'Careful')
plt.scatter(X[y_hc == 1, 0],X[y_hc == 1, 1], s = 50, c = 'blue', label = 'Standard')
plt.scatter(X[y_hc == 2, 0],X[y_hc == 2, 1], s = 50, c = 'yellow', label = 'Target')
plt.scatter(X[y_hc == 3, 0],X[y_hc == 3, 1], s = 50, c = 'magenta', label = 'Careless')
plt.scatter(X[y_hc == 4, 0],X[y_hc == 4, 1], s = 50, c = 'cyan', label = 'Sensible')
plt.title('Cluster of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend() 
plt.show()

