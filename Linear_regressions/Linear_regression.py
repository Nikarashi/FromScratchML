import numpy as np

class Analytical_Linear_Regression:

    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        X_train = np.insert(X_train,0,1,axis=1)
        self.wieghts = np.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)


    def predict(self, X_test):
        X_test = np.insert(X_test, 0, 1, axis=1)
        return X_test.dot(self.wieghts)
