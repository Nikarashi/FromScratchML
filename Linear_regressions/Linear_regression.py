import numpy as np

class Analytical_Linear_Regression:

    def fit(self, X_train, y_train):
        X_train = np.insert(X_train,0, 1, axis=1)
        weights = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

        
    def predict(self, X_test):
        y_pred = X_test.dot(self.weights)
        return y_pred
        


import numpy as np

# Пример данных
X_train = np.array([[1, 2], [2, 3], [3, 4]])
y_train = np.array([5, 7, 9])

X_test = np.array([[4, 5], [5, 6]])

# Создаем объект модели и обучаем ее
model = Analytical_Linear_Regression()
model.fit(X_train, y_train)

# Предсказываем на новых данных
predictions = model.predict(X_test)
print(predictions)