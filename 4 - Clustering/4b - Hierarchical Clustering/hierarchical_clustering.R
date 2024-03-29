# Hierarchical Clustering

# Importing the dataset
dataset <- read.csv('Mall_Customers.csv')
X <- dataset[4:5]

# Using the Dendrogram to find the optimal number of clusters
dendrogram <- hclust(dist(X, method = 'euclidean'), method = 'ward.D')
plot(dendrogram,
     main = paste('Dendogram'),
     xlab = 'Customers',
     ylab = 'Euclidean Distance')

# Fitting Hierarchical clustering to dataset
hc <- hclust(dist(X, method = 'euclidean'), method = 'ward.D')
y_hc <- cutree(hc, 5)


# Visualising the results
library(cluster)
clusplot(X,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of clients'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')
