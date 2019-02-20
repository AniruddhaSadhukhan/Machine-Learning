# Multi-Armed Bandit Problem : Upper Confidence Bound(UCB)

# Importing the dataset
dataset <- read.csv('Ads_CTR_Optimisation.csv')

# Implement UCB
N <- nrow(dataset)
d <- ncol(dataset)
numbers_of_selections <- integer(d)
sums_of_rewards <- integer(d)
upper_bound <- integer(d)
ads_selected <- integer()

for (n in 1:N)
{
    for (i in 1:d)
    {
        if (numbers_of_selections[i]>0)
        {
            average_reward <- sums_of_rewards[i]/numbers_of_selections[i]
            delta_i <- sqrt( ( 3 * log(n) )/(2 * numbers_of_selections[i]) )
            upper_bound[i] <- average_reward + delta_i
        }
        else
        {
            upper_bound[i] = 1e400
        }
    }
    ad = which.max(upper_bound)
    ads_selected <- append(ads_selected, ad)
    numbers_of_selections[ad] <- numbers_of_selections[ad] + 1
    sums_of_rewards[ad] <- sums_of_rewards[ad] + dataset[n,ad]
}

total_reward = sum(sums_of_rewards)

# Visualizing results
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of Ads selected',
     xlab = 'Ads',
     ylab = 'Number of times ad was clicked'
     )
