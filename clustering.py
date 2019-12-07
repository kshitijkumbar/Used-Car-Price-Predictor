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
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import LinearRegression

MAX_NUM_CLUSTERS = 10

def kmeans(filename):
    data = generateOneHotDataFrame(filename)
    X = pd.DataFrame(data.drop(['Price'],axis=1))
    y = pd.DataFrame(data['Price'])
    X_train, _, _, _ = train_test_split(X, y, test_size = 0.10, random_state = 0)
    for NUM_CLUSTERS in range(3, MAX_NUM_CLUSTERS + 1):
        kmeans = MiniBatchKMeans(n_clusters = NUM_CLUSTERS , random_state = 0)
        kmeans.fit(X_train)
        carClusterNumber = kmeans.predict(X)
        data['ClusterNumber'] = carClusterNumber
        totalSamples = 0
        weightedScores = 0
        print("---------------------------------------------")
        for clusterNumber in range(NUM_CLUSTERS):
            dataForThisCluster = data[(data['ClusterNumber'] == clusterNumber)]
            dataForThisCluster = pd.DataFrame(dataForThisCluster.drop(['ClusterNumber'],axis=1))
            numSamplesInCluster = len(dataForThisCluster.index)
            X_c = pd.DataFrame(dataForThisCluster.drop(['Price'],axis=1))
            y_c = pd.DataFrame(dataForThisCluster['Price'])
            X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size = 0.10, random_state = 0)
            model = LinearRegression()
            model.fit(X_train_c, y_train_c)
            score = model.score(X_test_c, y_test_c)
            print("The Score on the test set is {}".format(score))
            if ((score < 1.0) and (score > 0.6)):
                weightedScores += numSamplesInCluster*score
                totalSamples += numSamplesInCluster
            else:
                print("Problem with cluster number {} for total clusters of {}".format(clusterNumber, NUM_CLUSTERS))
            score = model.score(X_train_c, y_train_c)
            print("The Score on the train set is {}".format(score))
        pass
        print("Weighted score for {} clusters is {}".format(NUM_CLUSTERS, weightedScores/totalSamples))
        print("---------------------------------------------")
    pass
        
    
    # X = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    # print(kmeans.labels_)
    # kmeans.predict([[0, 0], [12, 3]])
    # print(kmeans.cluster_centers_)

def main():
    kmeans("CleanData/true_car_listings.csv")


if __name__ == '__main__':
    main()