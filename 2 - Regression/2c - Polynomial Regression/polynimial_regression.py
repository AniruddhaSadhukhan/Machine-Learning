#	Polynomial Regression 

#Importing the libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

#Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[: , 1:-1].values
y = dataset.iloc[: , -1].values

#Fitting Polynomial Regression to the Dataset
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree=4)
X_poly = poly_regressor.fit_transform(X)
poly_regressor.fit(X_poly,y)

from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(X_poly, y)

#Predicting test set results
y_pred = lin_regressor.predict(X_poly)
 
# Visualising Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_regressor.predict(poly_regressor.fit_transform(X_grid)),color = 'blue')
plt.title('Truth or Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#Predicting a new result
lin_regressor.predict(poly_regressor.fit_transform([[6.5]]))
