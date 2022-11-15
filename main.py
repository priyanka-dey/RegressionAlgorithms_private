from preprocessor import DataLoader
from logr import LogisticRegression 
from svm import SVM 
import pandas as pd 

#df = pd.read_csv("./diabetes.csv")
#df = pd.read_csv("./iris.csv") 
dl = DataLoader("./diabetes.csv") #df.iloc[:, :-1].values, df.iloc[:,-1].values)
X_train, X_test, y_train, y_test = dl.load_data()
logr = LogisticRegression()
logr.fit(X_train, y_train) 
print(logr.eval(X_test, y_test))  
svm = SVM()
svm.fit(X_train, y_train) 
print(svm.eval(X_test, y_test))  

