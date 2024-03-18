# -*- coding: utf-8 -*-
"""
AML-Project 
Song Popularity Prediction
Applied Machine Leaning 
CCDS-322
Created on Wed Feb  8 01:26:59 2023
"""
#Importing the basic librarires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import dataset
dataset = pd.read_csv('song_data.csv')
X = dataset.iloc[:, 4:5].values
Y = dataset.iloc[:, 1].values

#Spilt data 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state=0)

#Training the Simple Linear Regression model on the Training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the Test set results
Y_pred = regressor.predict(X_test)

#Visualising the Training set results
plt.rcParams['figure.figsize'] = [20, 12]
plt.scatter(X_train, Y_train, color = 'purple')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title('(Training set)')
plt.xlabel('Danceability')
plt.ylabel('Popularity')
plt.show()

#Visualising the Test set results
plt.rcParams['figure.figsize'] = [20, 12]
plt.scatter(X_test, Y_test, color = 'purple')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title('(Test set)')
plt.xlabel('Danceability')
plt.ylabel('Popularity')
plt.show()

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X, Y)

#Visualizing the Linear Regression results
def viz_linear():
    plt.rcParams['figure.figsize'] = [20, 12]
    plt.scatter(X, Y, color = 'purple')
    plt.plot(X, lin_reg.predict(X), color='red')
    plt.title('(Linear Regression)')
    plt.xlabel('Danceability')
    plt.ylabel('Popularity')
    plt.show()
    return
viz_linear()
