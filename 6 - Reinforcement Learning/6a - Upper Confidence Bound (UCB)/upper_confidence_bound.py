# Multi-Armed Bandit Problem : Upper Confidence Bound(UCB)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
N,d = dataset.shape
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
upper_bound = [0] * d
ads_selected = []

for n in range(N):
    for i in range(d):
        if numbers_of_selections[i]>0:
            average_reward = sums_of_rewards[i]/numbers_of_selections[i]
            delta_i = math.sqrt( ( 3 * math.log(n+1) )/(2 * numbers_of_selections[i]) )
            upper_bound[i] = average_reward + delta_i
        else:
            upper_bound[i] = 1e400
    ad = max(range(d), key = upper_bound.__getitem__)
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    sums_of_rewards[ad] += dataset.values[n,ad]
    
total_reward = sum(sums_of_rewards)

# Plotting Histogram of Ads selected
plt.hist(ads_selected)
plt.title('Histogram of Ads selected')
plt.xlabel('Ads')
plt.ylabel('Number of times ad was selected')
plt.show()