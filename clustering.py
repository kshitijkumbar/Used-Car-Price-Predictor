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
from sklearn.cluster import KMeans

def kmeans(filename):
    data = generateOneHotDataFrame(filename)
    X = pd.DataFrame(data.drop(['Price'],axis=1))
    y = pd.DataFrame(data['Price'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 1)
    kmeans = KMeans(n_clusters = 8, random_state = 0).fit(X_train)
    print(kmeans.predict(X_test))
    
    # X = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    # print(kmeans.labels_)
    # kmeans.predict([[0, 0], [12, 3]])
    # print(kmeans.cluster_centers_)

def main():
    kmeans("CleanData/true_car_listings.csv")


if __name__ == '__main__':
    main()