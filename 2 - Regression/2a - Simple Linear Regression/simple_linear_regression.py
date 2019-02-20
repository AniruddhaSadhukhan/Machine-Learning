#	Simple Linear Regression

# Importing the libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

# Import the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[: , :-1].values
y = dataset.iloc[: , 1].values

# Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting Test set result￼s
y_pred = regressor.predict(X_test)

# Visualizing Training set results
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience(Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Visualizing Test set results
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience(Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()