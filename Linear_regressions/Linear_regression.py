import numpy as np

class MatrixLinearRegression:

    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        X_train = np.insert(X_train,0,1,axis=1)
        self.weights = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)


    def predict(self, X_test):
        X_test = np.insert(X_test, 0, 1, axis=1)
        return X_test.dot(self.weights)


class LinearRegressionGD :

    def __init__(self, lr=0.01):
        self.lr = lr

    def fit(self, X_train, y_train):
        X_train = np.insert(X_train, 0, 1, axis=1)
        self.weights = self.__gradient_descent(X_train, y_train)

    def predict(self, X_test):
        X_test = np.insert(X_test, 0, 1, axis=1)
        return X_test.dot(self.weights)

    def __gradient_descent(self, X_train, y_train):
        self.weights = np.random.normal(size=X_train.shape[1])
        n = X_train.shape[0]
        diff = 10000
        err = 1e10

        while np.linalg.norm(err) > diff:
            xw = X_train.dot(self.weights)
            err = xw-y_train
            gradient = 2 * X_train.T.dot(err)/n
            self.weights -= self.lr * gradient
        return self.weights
