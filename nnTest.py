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
import lightgbm as lgb
# load dataset
# SHAPE = 0
# def baseline_model(SHAPE):
#     # create model
#     def bm():
#         model = Sequential()
#         print(f"{SHAPE}")
#         model.add(Dense(SHAPE, input_dim=SHAPE, kernel_initializer='normal', activation='relu'))
#         model.add(Dense(1250, kernel_initializer='normal', activation='relu'))
#         # model.add(Dense(10, kernel_initializer='normal', activation='relu'))
#         model.add(Dense(1, kernel_initializer='normal'))
        
#         # Compile model
#         model.compile(loss='mean_squared_error', optimizer='adam')
#         return model
#     return bm
#     # evaluate model


def linearRegressionOneHot(filename):
    data = generateOneHotDataFrame(filename)
    X = pd.DataFrame(data.drop(['Price'],axis=1))
    y = pd.DataFrame(data['Price'])
    SHAPE = (np.shape(X)[1])
    print(SHAPE)
    # print(X.head)
    # model =    XGBRegressor(verbosity = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 1)
    print("Now fitting")
    print(np.shape(X_train)[1])
    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train,categorical_feature = [3,4,5] , verbose = 1)
    score = model.score(X_test, y_test)
    print(f"The Score on the test set is {score}")
    score = model.score(X_train, y_train)
    print(f"The Score on the train set is {score}")

def main():
    linearRegressionOneHot("CleanData/true_car_listings.csv")


if __name__ == '__main__':
    main()
