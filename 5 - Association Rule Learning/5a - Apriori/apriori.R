# Market Basket Analysis - Apriori

# Importing the dataset
# install.packages('arules')
library(arules)
# dataset <- read.csv('Market_Basket.csv', header = FALSE)
dataset <- read.transactions('Market_Basket.csv', 
                             sep = ',', 
                             rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 50)    
     

# Training Apriori on the dataset
rules <- apriori(data = dataset,
                 parameter = list(support = 0.004, confidence = 0.2))

# Visualising the results
inspect(sort(rules, by = 'lift')[1:10])
