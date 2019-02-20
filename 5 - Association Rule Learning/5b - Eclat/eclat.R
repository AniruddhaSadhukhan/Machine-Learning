# Market Basket Analysis - Eclat

# Importing the dataset
# install.packages('arules')
library(arules)
# dataset <- read.csv('Market_Basket.csv', header = FALSE)
dataset <- read.transactions('Market_Basket.csv', 
                             sep = ',', 
                             rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 20)    


# Training Eclat on the dataset
rules <- eclat(data = dataset,
                 parameter = list(support = 0.004, minlen = 2))

# Visualising the results
inspect(sort(rules, by  = 'support')[1:10])
