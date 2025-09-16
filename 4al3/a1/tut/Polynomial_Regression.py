
# Author: Swati Mishra
# Created: Sep 9, 2024
# License: MIT License
# Purpose: This python code includes Polynomial Regression 

# Usage: python polynomial_regression.py

# Dependencies: None
# Python Version: 3.6+

# Modification History:
# - Version 1 - added polynomial regression implementation

# References:
# - https://www.python.org/dev/peps/pep-0008/
# - Python Documentation

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import data
data = pd.read_csv("gdp-vs-happiness.csv")
data2 = data[:2000]
#drop columns that will not be used
by_year = (data[data['Year']==2018]).drop(columns=["Continent","Population (historical estimates)","Code"])
# remove missing values from columns 
df = by_year[(by_year['Cantril ladder score'].notna()) & (by_year['GDP per capita, PPP (constant 2017 international $)']).notna()]

#create np.array for gdp and happiness where happiness score is above 4.5
happiness=[]
gdp=[]
for row in df.iterrows():
    if row[1]['Cantril ladder score']:
        happiness.append(row[1]['Cantril ladder score'])
        gdp.append(row[1]['GDP per capita, PPP (constant 2017 international $)'])

class polynomial_regression():
 
    def __init__(self,x_:list,y_:list) -> None:

        self.input = np.array(x_)
        self.target = np.array(y_)

    def preprocess(self,):

        #normalize the values
        hmean = np.mean(self.input)
        hstd = np.std(self.input)
        x_train = (self.input - hmean)/hstd

        #arrange in matrix format
        X = np.column_stack([x_train,x_train**2,x_train**3])

        #normalize the values
        gmean = np.mean(self.target)
        gstd = np.std(self.target)
        y_train = (self.target - gmean)/gstd

        #arrange in matrix format
        Y = (np.array([y_train])).T

        return X, Y

    def train(self, X, Y):
        #compute and return beta
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    
    def predict(self, x_test,beta):

        # fit the polynomial
        x_poly = np.stack([x_test, x_test**2, x_test**3], axis=1)

        #predict the y points using x points
        return np.sum(x_poly.dot(beta),axis=1)

#instantiate the linear_regression class  
poly_reg = polynomial_regression(happiness,gdp)

# preprocess the inputs
X_,Y_ = poly_reg.preprocess()

#compute beta
beta = poly_reg.train(X_,Y_)

# generate 10 random set of points between -2 and 2
x_sample = np.linspace(-2, 2, 10)

# use the computed beta for prediction
Y_test = poly_reg.predict(x_sample,beta)

# access the 0st column
X = X_[...,0].ravel()

# below code displays the predicted values

#set the plot and plot size
fig, ax = plt.subplots()
fig.set_size_inches((15,8))

# display the X and Y points
ax.scatter(X,Y_)

#display the line predicted by beta and X
ax.plot(x_sample,Y_test,color='r')

#set the x-labels
ax.set_xlabel("Happiness")

#set the x-labels
ax.set_ylabel("GDP per capita")

#set the title
ax.set_title("Cantril Ladder Score vs GDP per capita of countries (2018)")

#show the plot
plt.show()
