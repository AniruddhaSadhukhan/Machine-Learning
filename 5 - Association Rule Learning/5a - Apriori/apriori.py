# Market Basket Analysis - Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Market_Basket.csv',header = None)

transactions = []
for i in range(dataset.shape[0]):
    transactions.append([str(dataset.values[i,j]) for j in range(dataset.shape[1])])


# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, 
                min_support = 0.004, 
                min_confidence = 0.2,
                min_lift = 3, 
                min_length = 2)

# Visualising the result
results = list(rules)
for i in range(10):
    print(results[i])




