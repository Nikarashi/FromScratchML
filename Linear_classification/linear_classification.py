import numpy as np
class LogisticLoss:
    def __call__(self, w, x, y):
        m = y*(w.T.dot(x))
        return np.mean(np.log(1+np.exp(-m)))
    def grad(self,x,y,w):
        m = y*(w.T.dot(x))
        sigmoid = 1/(1+np.exp(m))
        return - np.mean(y.reshape(-1,1)*x*sigmoid)

class LinearClassification:
    def __init__(self, lr=0.01, max_iter=100):
        self.lr = lr
        self.max_iter = 100
        self.loss = LogisticLoss()

    def fit(self, X_train, y_train):
        X_train = np.insert(X_train, 0, 1, axis=1)
        self.weights = self.__gradient_desc(X_train, y_train)

    def predict(self, X_test):
        X_test = np.insert(X_test, 0, 1, axis=1)
        return np.sign(X_test.dot(self.weights))

    def __gradient_desc(self, X_train,y_train):
        self.weights = np.random.normal(size=X_train.shape[1])

        for _ in range(self.max_iter):
            gradient = self.loss.grad(X_train,y_train,self.weights)
            self.weights -= self.lr*gradient
        return self.weights

X_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_train = np.array([-1, 1, 1, -1]) 

model = LinearClassification(lr=0.1, max_iter=1000)
model.fit(X_train, y_train)  

X_test = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
predictions = model.predict(X_test)
print(predictions)


        
