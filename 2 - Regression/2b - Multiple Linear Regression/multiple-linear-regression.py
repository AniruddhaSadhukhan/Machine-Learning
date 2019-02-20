#	Multiple Linear Regression 

#Importing the libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

#Import the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1].values


#Encoding Catagorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,-1] = labelencoder_X.fit_transform(X[:,-1])
onehotencoder = OneHotEncoder(categorical_features=[-1])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:, 1:]

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting test set results
y_pred = regressor.predict(X_test)


#==================================================================#
#   Backward Elimination Method for Multiple Linear Regression     #
#==================================================================#

#   Building the optimal model using Backward Elimination manually
"""import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values= X, axis = 1)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#Removing x2 having p value 0.990
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#Removing x1 having p value 0.940
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#Removing x2 having p value 0.602
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#Removing x2 having p value 0.060
X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()"""


#   Building the optimal model using Backward Elimination automatically
from statsmodels.tools.tools import add_constant
X_opt = add_constant(X,prepend = True)
import statsmodels.formula.api as sm
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
initial_summary = regressor_OLS.summary()
pvalues = regressor_OLS.pvalues
rsquared_adj = regressor_OLS.rsquared_adj


while max(pvalues) > 0.05:
    max_index = np.argmax(pvalues)
    X_opt = np.delete(X_opt, max_index, axis=1)
    new_regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    
    if rsquared_adj > new_regressor_OLS.rsquared_adj:
        break
   
    regressor_OLS = new_regressor_OLS
    pvalues = regressor_OLS.pvalues
    rsquared_adj = regressor_OLS.rsquared_adj

final_summary = regressor_OLS.summary()

#Splitting the dataset into training and test set
X_opt_train, X_opt_test, y_opt_train, y_opt_test = train_test_split(X_opt,y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression to the Training set
regressor_OLS = LinearRegression()
regressor_OLS.fit(X_opt_train,y_opt_train)

#Predicting test set results
y_opt_pred = regressor_OLS.predict(X_opt_test)














