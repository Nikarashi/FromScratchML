import numpy as np

class Analytical_Linear_Regression:

    def fit(self, X_train, y_train):
        X_train = np.insert(X_train,0, 1, axis=1)
        weights = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

        
    def predict(self, X_test):
        y_pred = X_test.dot(self.weights)
        return y_pred
        
