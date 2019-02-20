# Churn Detection : Artificial Neural Network

# Installing Theano
# conda install theano

# Installing Keras
# pip install keras

# Configure Keras to use Theno backend
# Update ~/.keras/keras.json to change backend to "theano" from "tensorflow"

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#Encoding Catagorical data
#Encoding Independent Variable
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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Tip : Nodes in hidden layer(generally) = (Input nodes + Output nodes)/2 = (11+1)/2 =6
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6,
                     kernel_initializer= 'uniform',
                     activation = 'relu',
                     input_dim = 11)) 
# relu : rectifier actifation function

# Adding the second hidden layer
classifier.add(Dense(units = 6,
                     kernel_initializer= 'uniform',
                     activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1,
                     kernel_initializer= 'uniform',
                     activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])
# adam : an efficient stochastic gradient decent algorithm

# Fitting the ANN to the Training set
classifier.fit(x = X_train,
               y = y_train,
               batch_size = 10,
               epochs = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting testset result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

