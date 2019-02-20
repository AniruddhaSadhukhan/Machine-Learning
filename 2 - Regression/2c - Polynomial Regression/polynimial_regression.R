# Polynomial Regression

#Import the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Fitting Polynomial Regression to the dataset
    #  # Manual method
    # dataset$Level2 = dataset$Level^2
    # dataset$Level3 = dataset$Level^3
    # dataset$Level4 = dataset$Level^4
    # 
    # regressor = lm(formula = Salary ~ . ,
    #                data = dataset)

 # Automatic Method
regressor = lm(formula = Salary ~ poly(Level,4) ,
                data = dataset)

summary(regressor)

# Predicting Test Set results
y_pred = predict(regressor, newdata = dataset)

# Visualising Polynomial Regression results
library(ggplot2)
ggplot() + 
    geom_point(aes(x = dataset$Level, y = dataset$Salary), 
               color = 'red') +
    geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
              color = 'blue') +
    ggtitle('Truth or Bluff') +
    xlab('Position Label') +
    ylab('Salary')


# Visualising the Polynomial Regression Model results (for higher resolution and smoother curve)
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

# Predicting a new result
x = 6.5 
y_new_pred = predict(regressor, newdata = data.frame(Level = x))
