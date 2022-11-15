from sklearn.svm import SVC 
from sklearn.metrics import classification_report, f1_score

class SVM:
	def __init__(self, kernel = 'rbf', C = 1.0):	
		self.kernel = kernel
		self.C = C 
		self.classifier = None

	def fit(self, X_train, y_train): 	
		classifier = SVC(C=self.C, kernel=self.kernel) 
		classifier.fit(X_train, y_train)
		self.classifier = classifier

	def eval(self, X_test, y_test):
		y_pred = self.classifier.predict(X_test)
		#print(classification_report(y_test, y_pred))
		return f1_score(y_test, y_pred, average='macro')

