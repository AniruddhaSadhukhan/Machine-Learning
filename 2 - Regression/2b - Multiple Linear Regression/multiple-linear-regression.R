# Multiple Linear Regression

#Import the dataset
dataset = read.csv('50_Startups.csv')

#Encoding Catagorical data
dataset$State =  factor(dataset$State,
                        levels = c('New York','California','Florida'),
                        labels = c(1,2,3))

#Splitting the dataset into training and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit,SplitRatio = 0.8)
training_set = subset(dataset,split == TRUE)
test_set = subset(dataset,split == FALSE)

# Fitting Multiple Linear Regression to the Training set
#regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State ,
#               data = training_set)
regressor = lm(formula = Profit ~ . ,
               data = training_set)
summary(regressor)

# Predicting Test Set results
y_pred = predict(regressor,newdata = test_set)


#===========================================================#
#   Building the optimal model using Backward Elimination   #
#===========================================================#
# Manually building optimum regressor
regressor_optimum = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State ,
              data = dataset)
summary(regressor_optimum)
# Removing State2 & State3
regressor_optimum = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor_optimum)
# Removing Administration
regressor_optimum = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor_optimum)
# Removing Marketing Spend
regressor_optimum = lm(formula = Profit ~ R.D.Spend ,
               data = dataset)
summary(regressor_optimum)

# Automatically building optimum regressor using AIC instead of p-value
    # regressor_optimum = lm(formula = Profit~.,
    #                        data = dataset)
    # initial_summary = summary(regressor_optimum)
    # regressor_optimum = step(regressor_optimum, direction = "backward")
    # final_summary = summary(regressor_optimum)


# Predicting Test Set results
y_opt_pred = predict(regressor_optimum,newdata = test_set)












