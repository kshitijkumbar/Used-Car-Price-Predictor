#!usr/bin/env/python3
'''
Gets the pandas dataframe and performs linear regression

'''
from dataClean import *
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression

def linearRegressionOneHot(filename):
    data = generateOneHotDataFrame(filename)
    X = pd.DataFrame(data.drop(['Price'],axis=1))
    y = pd.DataFrame(data['Price'])
    # print(X.head)
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 1)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"The Score on the test set is {score}")
    score = model.score(X_train, y_train)
    print(f"The Score on the train set is {score}")
    #scores = []
    # kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    # for i, (train, test) in enumerate(kfold.split(X, y)):
    #     model.fit(X.iloc[train,:], y.iloc[train,:])
    #     score = model.score(X.iloc[test,:], y.iloc[test,:])
    #     scores.append(score)
    # print(scores)

def main():
    linearRegressionOneHot("CleanData/true_car_listings.csv")


if __name__ == '__main__':
    main()