import numpy as np
# Комментарий для проверки

class KnnClassifire:

    def __init__(self, k=3, metric='Euclidean'):
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        dist = self.__calculating_distance(X)

        return self.predict_labels(dist, self.k)

    def __calculating_distance(self, X):

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        distance = np.zeros(num_test, num_train)

        if self.metric == 'Euclidean':
            sq_train = np.sum(self.X_train**2, axis=1)
            sq_test = np.sum(X**2, axis=1)[:, np.newaxis]
            distance = np.sqrt(sq_test + sq_train - 2 * X.dot(self.X_train.T))
        return distance
    
    def __predict_labels(dist, k):
        pass

