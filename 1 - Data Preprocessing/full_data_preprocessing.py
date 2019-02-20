#	Data preprocessing

#Importing the libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

#Import the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:,-2:])
X[:,-2:] = imputer.transform(X[:,-2:])
print(X)

#Encoding Catagorical data
#Encoding Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
#Encoding Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



