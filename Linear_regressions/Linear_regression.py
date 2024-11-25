import numpy as np

class Analytical_Linear_Regression:

    def fit(self, X_train, y_train):
        X_train = np.insert(X_train,0, 1, axis=1)
        self.weights = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
        self.bias = self.weights[0]
        self.weights = self.weights[1]
        
    def predict(self, X_test):
        y_pred = X_test.dot(self.weights) + self.bias
        return y_pred
        
