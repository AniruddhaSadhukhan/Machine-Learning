# Decision Tree Regression

#Import the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Fitting regression model to the dataset
library(rpart)

regressor = rpart(formula = Salary ~ . ,
                data = dataset,
                control = rpart.control(minsplit = 1))

# Predicting new test results
test_levels = c(6.5,7.5)
y_pred = predict(regressor, newdata = data.frame(Level = test_levels))


# Visualising the regression Model results (for higher resolution and smoother curve)
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
    geom_point(aes(x = dataset$Level, y = dataset$Salary),
               colour = 'red') +
    geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
              colour = 'blue') +
    ggtitle('Truth or Bluff (Decision Tree Model)') +
    xlab('Level') +
    ylab('Salary')


