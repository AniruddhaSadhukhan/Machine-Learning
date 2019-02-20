# Multi-Armed Bandit Problem : Thompson Sampling

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling
N,d = dataset.shape
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
random_beta = [0] * d
ads_selected = []
total_reward = 0

for n in range(N):
    for i in range(d):
        random_beta[i] = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
    ad = max(range(d), key = random_beta.__getitem__)
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    if reward:
        numbers_of_rewards_1[ad] +=1
    else:
        numbers_of_rewards_0[ad] +=1
    total_reward += reward
    
# Plotting Histogram of Ads selected
plt.hist(ads_selected)
plt.title('Histogram of Ads selected')
plt.xlabel('Ads')
plt.ylabel('Number of times ad was selected')
plt.show()
