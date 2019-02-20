# Support Vector Regression

#Import the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Fitting SVR to the dataset
# install.packages('e1071')
library(e1071)

regressor = svm(formula = Salary ~ . ,
               data = dataset,
               type = 'eps-regression')

# Predicting new test results
test_levels = c(6.5,7.5)
y_pred = predict(regressor, newdata = data.frame(Level = test_levels))

# Visualising SVR results
library(ggplot2)
ggplot() + 
    geom_point(aes(x = dataset$Level, y = dataset$Salary), 
               color = 'red') +
    geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
              color = 'blue') +
    ggtitle('Truth or Bluff') +
    xlab('Position Label') +
    ylab('Salary')


# Visualising the SVR Model results (for higher resolution and smoother curve)
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
    geom_point(aes(x = dataset$Level, y = dataset$Salary),
               colour = 'red') +
    geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
              colour = 'blue') +
    ggtitle('Truth or Bluff (Polynomial Regression Model)') +
    xlab('Level') +
    ylab('Salary')


