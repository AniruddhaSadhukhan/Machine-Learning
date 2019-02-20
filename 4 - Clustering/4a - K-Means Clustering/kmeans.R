# K-means Clustering

# Importing the dataset
dataset <- read.csv('Mall_Customers.csv')
X <- dataset[4:5]

# Using the elbow method to find the optimal number of clusters
set.seed(6)
WCSS <- vector()
for (i in 1:10) WCSS[i] <- sum(kmeans(X,i)$withinss)
plot(1:10, WCSS, type = 'b', 
     main = paste('The Elbow Method'),
     xlab = 'Number of clusters',
     ylab = 'WCSS')

# Applying k-means to dataset
set.seed(29)
kmeans <- kmeans(X,5, iter.max = 300,nstart = 10)
y_kmeans = kmeans$cluster

# Visualising the results
library(cluster)
clusplot(X,
         y_kmeans,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of clients'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')
