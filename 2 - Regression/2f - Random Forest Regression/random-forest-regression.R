#           Random Forest Regression
#Import the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Fitting regression model to the dataset
# install.packages('randomForest')
library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[1], y = dataset$Salary, ntree = 500)

# Predicting new test results
test_levels = c(6.5,7.5)
y_pred = predict(regressor, newdata = data.frame(Level = test_levels))

# Visualising the regression Model results (for higher resolution and smoother curve)
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
    geom_point(aes(x = dataset$Level, y = dataset$Salary),
               colour = 'red') +
    geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
              colour = 'blue') +
    ggtitle('Truth or Bluff (Random Forest Regression Model)') +
    xlab('Level') +
    ylab('Salary')


