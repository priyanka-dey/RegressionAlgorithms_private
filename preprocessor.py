import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataLoader:
	def __init__(self, file_path, split_ratio=0.25):
		df = pd.read_csv(file_path) 
		X, Y = df.iloc[:, :-1].values, df.iloc[:,-1].values  
		self.split_ratio = split_ratio
		self.X = X
		self.Y = Y

	def load_data(self): 
		X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size = self.split_ratio, random_state = 0)
		sc = StandardScaler()
		X_train = sc.fit_transform(X_train)
		X_test = sc.transform(X_test)
		return X_train, X_test, y_train, y_test 
