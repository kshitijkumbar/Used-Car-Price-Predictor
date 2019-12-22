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
#     fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
#     n_bins = 100
# # We can set the number of bins with the `bins` kwarg
#     axs[0].hist(data['Mileage'], bins=n_bins)
#     axs[1].hist(data['Price'], bins=n_bins)
#     axs[2].hist(data['Year'], bins=n_bins)
    data.hist(bins =100,layout=(2,3),sharey = True)
    plt.show()
    std_dev = 3
    data = data[(np.abs(stats.zscore(data[['Mileage','Price']])) < float(std_dev)).all(axis=1)]
    data.hist(bins =100,layout=(2,3))
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
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.iloc[0:500000,:]
    data = pd.concat([data,pd.get_dummies(data['State'])],axis=1).drop(['State'],axis=1)
    data = pd.concat([data,pd.get_dummies(data['Make'])],axis=1).drop(['Make'],axis=1)
    data = pd.concat([data,pd.get_dummies(data['Model'])],axis=1).drop(['Model'],axis=1)
    return data

def plotsForLinearity(filename):
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
    std_dev = 3
    data = data[(np.abs(stats.zscore(data[['Mileage','Price']])) < float(std_dev)).all(axis=1)]
    onlyAccord = data[data['Model'].str.contains("accord")]
    plt.figure()
    plt.scatter(onlyAccord['Year'].astype(int), onlyAccord['Price'], marker='x') 
    plt.xlabel("Year")
    plt.ylabel("Price")
    plt.savefig("priceVsYear.png")
    plt.figure()
    plt.scatter(onlyAccord['Mileage'].astype(int), onlyAccord['Price'],marker='x') 
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.savefig("priceVsMileage.png")


def main():
    #data_processor('RawData/true_car_listings.csv')
    plotsForLinearity('RawData/true_car_listings.csv')

if __name__ == '__main__':
    main()