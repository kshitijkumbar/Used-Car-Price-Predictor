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
from sklearn.svm import SVR
import pickle


def main():
    data = generateOneHotDataFrame("CleanData/true_car_listings.csv")
    X = pd.DataFrame(data.drop(['Price'],axis=1))
    y = pd.DataFrame(data['Price'])
    y = np.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)
    clf = SVR(kernel='poly', degree=3, max_iter=100000, verbose=True)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f"The Score is {score}")
    file_name = "SVRModel"
    pickle.dump(clf, open(file_name, 'wb'))


if __name__ == '__main__':
    main()
