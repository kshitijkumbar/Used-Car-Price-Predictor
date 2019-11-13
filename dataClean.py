import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing

def data_processor(filename):
    """
    Process data from data file

    Used to preprocess data from given datafile for further processing according to requirements
    Strips all lettercase and special characters between strings


    Parameters:
    filename (string) : Name of the datafile(CSV) (Eg. "data.csv")
    
    Returns:

    data (Pandas Dataframe) : Dataframe containing the processed data

    """
    data = pd.read_csv(filename,low_memory=False)
    pd.set_option('display.max_columns', None)
    print(f"Are any entries missing? : {data.isnull().values.any()}")
    data['Vin'] = data['Vin'].str.lower()
    data['City'] = data['City'].str.lower()
    data['City'] = data['City'].str.replace(" ","")
    data['State'] = data['State'].str.lower()
    data['Make'] = data['Make'].str.lower()
    data['Model'] = data['Model'].str.lower()
    data['Model'] = data['Model'].str.replace(" ","")
    data['Model'] = data['Model'].str.replace("-","")
    data.hist()
    plt.show()
    std_dev = 3
    data = data[(np.abs(stats.zscore(data[['Mileage','Price']])) < float(std_dev)).all(axis=1)]
    data.hist()
    plt.show()
    # Drop City and Vin#
    data = data.drop(['Vin'],axis=1)
    data = data.drop(['City'],axis=1)
    savePath = "CleanData/" + filename.split('/')[1]
    data.to_csv(savePath, encoding='utf-8', index=False)
    print("Done cleanup.")
    return


def generateOneHotDataFrame(filename):
    data = pd.read_csv(filename)
    data = data.iloc[0:500000,:]
    data = pd.concat([data,pd.get_dummies(data['State'])],axis=1).drop(['State'],axis=1)
    data = pd.concat([data,pd.get_dummies(data['Make'])],axis=1).drop(['Make'],axis=1)
    data = pd.concat([data,pd.get_dummies(data['Model'])],axis=1).drop(['Model'],axis=1)
    return data


def main():
    data_processor('RawData/true_car_listings.csv')

if __name__ == '__main__':
    main()