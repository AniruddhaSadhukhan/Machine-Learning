# Natural Language Processing 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', 
                      delimiter = '\t', 
                      quoting = 3)


# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(dataset.shape[0]):
    # Keeping only words
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    
    # Removing irrelevant words like articles, prepositions etc. and
    # Stemming: convert to the root word Eg- loved -> love
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# Create Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#********************* Use any one classifier *********************
# Fitting the Classifier to the Training set : Naive Bayes 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)
#------------------------------------------------------------------
# Fitting the Classifier to the Training set : Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',
                                    random_state=0)
classifier.fit(X_train,y_train)
#------------------------------------------------------------------
# Fitting the Classifier to the Training set : Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10,
                                    criterion = 'entropy',
                                    random_state = 0)
classifier.fit(X_train,y_train)
#*****************************************************************

# Predicting testset result
y_pred = classifier.predict(X_test)

# Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# Performance
TN, FP, FN, TP = cm.ravel()
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN) 
F1_Score = 2 * Precision * Recall / (Precision + Recall)

