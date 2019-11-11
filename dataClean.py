import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
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
	data = pd.read_csv(filename)
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
	print(f"Mean Mileage: {np.mean(data.Mileage.values)}, Max Mileage : {np.max(data.Mileage.values)}, Min Mileage : {np.min(data.Mileage.values)}")
	# plt.plot(data.Mileage.values-np.mean(data.Mileage.values))
	# plt.show()
	return data




def main():
	data_processor('Data/true_car_listings.csv')


if __name__ == '__main__':
	main()