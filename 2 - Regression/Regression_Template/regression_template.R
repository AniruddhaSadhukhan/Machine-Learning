# Regression Template

# Importing the dataset
dataset = read.csv('Data.csv')
# dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting the Regression Model to the dataset
# Create your regressor here

# Predicting a new result
y_pred = predict(regressor, test_set)

# Visualising the Regression Model results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$X, y = dataset$Y),
             colour = 'red') +
  geom_line(aes(x = dataset$X, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Title (Regression Model)') +
  xlab('X') +
  ylab('Y')

# Visualising the Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$X), max(dataset$X), 0.1)
ggplot() +
  geom_point(aes(x = dataset$X, y = dataset$Y),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(X = x_grid))),
            colour = 'blue') +
  ggtitle('Table (Regression Model)') +
  xlab('X') +
  ylab('Y')