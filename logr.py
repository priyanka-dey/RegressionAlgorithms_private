from sklearn.linear_model import LogisticRegression as logistic_regression
from sklearn.metrics import classification_report, f1_score

class LogisticRegression:
	def __init__(self, solver = 'lbfgs', C = 1.0):	
		self.solver = solver
		self.C = C 
		self.classifier = None

	def get_details(self): 
		return ("LogR", "C = " + str(self.C) + ", Solver = '" + self.solver + "'") 

	def fit(self, X_train, y_train): 	
		classifier = logistic_regression(random_state = 0, C=self.C, solver=self.solver, multi_class='auto')
		classifier.fit(X_train, y_train)
		self.classifier = classifier

	def eval(self, X_test, y_test):
		y_pred = self.classifier.predict(X_test)
		#print(classification_report(y_test, y_pred))
		return f1_score(y_test, y_pred, average='macro')

