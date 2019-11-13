import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
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
	# plt.plot(histo)
	# plt.show()
	# print(data[(data.Mileage == 5)])
	data=data.iloc[0:250000,:]
	print(len(data.Model.unique())+len(data.Make.unique())+len(data.State.unique())+len(data.City.unique()))
	
	print(f"Mean Mileage: {np.mean(data.Mileage.values)}, Max Mileage : {np.max(data.Mileage.values)}, Min Mileage : {np.min(data.Mileage.values)}")
	data = pd.concat([data,pd.get_dummies(data['State'])],axis=1).drop(['State'],axis=1)
	data = pd.concat([data,pd.get_dummies(data['City'])],axis=1).drop(['City'],axis=1)
	data = pd.concat([data,pd.get_dummies(data['Make'])],axis=1).drop(['Make'],axis=1)
	data = pd.concat([data,pd.get_dummies(data['Model'])],axis=1).drop(['Model'],axis=1)
	data = data.drop(['Vin'],axis=1)
	print((np.shape(data)))




	# plt.plot(data.Mileage.values-np.mean(data.Mileage.values))
	# plt.show()
	X = pd.DataFrame(data.drop(['Price'],axis=1))
	y = pd.DataFrame(data['Price'])
	# print(X.head)
	model = LinearRegression()
	scores = []
	kfold = KFold(n_splits=3, shuffle=True, random_state=42)
	for i, (train, test) in enumerate(kfold.split(X, y)):
		model.fit(X.iloc[train,:], y.iloc[train,:])
		score = model.score(X.iloc[test,:], y.iloc[test,:])
		scores.append(score)
	print(scores)
	return data




def main():
	data_processor('Data/true_car_listings.csv')


if __name__ == '__main__':
	main()