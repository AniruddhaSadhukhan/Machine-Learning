# Multi-Armed Bandit Problem : Thompson Sampling

# Importing the dataset
dataset <- read.csv('Ads_CTR_Optimisation.csv')

# Implement Thompson Sampling
N <- nrow(dataset)
d <- ncol(dataset)
numbers_of_rewards_1 <- integer(d)
numbers_of_rewards_0 <- integer(d)
random_beta <- integer(d)
ads_selected <- integer()
total_reward <- 0

for (n in 1:N)
{
    for (i in 1:d)
    {
        random_beta[i] = rbeta(n = 1, 
                               shape1 = numbers_of_rewards_1[i] + 1, 
                               shape2 = numbers_of_rewards_0[i] + 1)
    }
    ad = which.max(random_beta)
    ads_selected <- append(ads_selected, ad)
    reward <- dataset[n,ad]
    if (reward==1)
        numbers_of_rewards_1[ad] <- numbers_of_rewards_1[ad] + 1
    else
        numbers_of_rewards_0[ad] <- numbers_of_rewards_0[ad] + 1
    total_reward <- total_reward + reward
}


# Visualizing results
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of Ads selected',
     xlab = 'Ads',
     ylab = 'Number of times ad was clicked'
)
