# Churn Detection : XGBoost

# Install XGBoost
# conda install -c anaconda py-xgboost


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#Encoding Catagorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy variable trap
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Part 2 - Fitting XGBoost to training set

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)

# Part 3 - Making the predictions and evaluating the model

# Predicting testset result
y_pred = classifier.predict(X_test)

# Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# Applying k-fold Cross Vadidation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, 
                             X = X_train, 
                             y = y_train,
                             cv = 10)
mean_accuracy = accuracies.mean()
standard_deviation_of_accuracy = accuracies.std()

