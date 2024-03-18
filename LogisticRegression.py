# -*- coding: utf-8 -*-
"""
AML-Project 
Song Popularity Prediction
Applied Machine Leaning 
CCDS-322
Created on Thu Feb  9 17:49:52 2023
"""
#Importing the basic librarires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Import the dataset
dataset = pd.read_csv('song_data.csv')
dataset=dataset.drop("song_name", axis=1)
X = dataset[dataset.columns.difference(['song_popularity'])]
Y = dataset['song_popularity']


from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.feature_selection import chi2,mutual_info_classif,f_regression,f_classif
y_log = pd.cut(x=dataset["song_popularity"], bins=[-1,42,100], labels=[0,1]) 
X_log = SelectKBest(f_classif, k=13).fit_transform(X, y_log)

#Spilt data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_log, y_log, test_size=0.30, random_state=42)

#Training the Simple Linear Regression model on the Training set 
from sklearn.linear_model import LogisticRegression, LinearRegression
log_clf = LogisticRegression(random_state=42).fit(X_train, y_train)
from joblib import dump
dump(log_clf, "LogisticRegression")

#Predicting the Test set results
Y_pred = log_clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
cr = classification_report(y_test, Y_pred)
print(cr)

#Confusion Matrix
cm = confusion_matrix(y_test, Y_pred)
print(cm)

#Model Evaluation
from sklearn import metrics
from matplotlib import pyplot
Y_pred = log_clf.predict_proba(X_test)[::,1]
fpr, tpr,thresholds = metrics.roc_curve(y_test, Y_pred)

#Create ROC curve
np.save("fpr_LR", fpr)
np.save("tpr_LR", tpr)
#Plot the ROC curve for the model
pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
#Axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
#Show the plot
pyplot.show()

#Define metrics
Y_pred = log_clf.predict_proba(X_test)[::,1]
fpr, tpr,thresholds= metrics.roc_curve(y_test,  Y_pred)
auc = metrics.roc_auc_score(y_test, Y_pred)
#Plot the AUC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
#Axis labels
plt.ylabel('Recall')
plt.xlabel('Precision')
plt.legend(loc=4)
# Show the plot
plt.show()



