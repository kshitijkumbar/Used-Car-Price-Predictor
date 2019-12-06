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
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
# load dataset

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
    # evaluate model


def linearRegressionOneHot(filename):
    data = generateOneHotDataFrame(filename)
    X = pd.DataFrame(data.drop(['Price'],axis=1))
    y = pd.DataFrame(data['Price'])
    # print(X.head)
    # model =    XGBRegressor(verbosity = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 1)
    print("Now fitting")
    model = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)

    # kfold = KFold(n_splits=10)
    # results = cross_val_score(estimator, X, Y, cv=kfold)
    # print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
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
