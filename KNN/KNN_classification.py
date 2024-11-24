import numpy as np
class KnnClassifire:

    def __init__(self, k=3, metric='Euclidean'):
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        dist = self.__calculating_distance(X)

        return self.predict_labels()

    def __calculating_distance(self, X):

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        self.distance = np.zeros((num_test, num_train))

        if self.metric == 'Euclidean':
            sq_train = np.sum(self.X_train**2, axis=1)
            sq_test = np.sum(X**2, axis=1)[:, np.newaxis]
            self.distance = np.sqrt(sq_test + sq_train - 2 * np.dot(X, self.X_train.T))
        return self.distance
    
    def predict_labels(self):
        n_test = self.distance.shape[0]
        y_pred = np.zeros((n_test))

        for i in range(n_test):
            sorting_k = np.argsort(self.distance[i,])[:self.k]
            closest_y = self.y_train[sorting_k]
            label = np.argmax(np.bincount(closest_y))
            y_pred[i] = label

        return y_pred




        

