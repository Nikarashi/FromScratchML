import numpy as np
class LogisticLoss:
    def __call__(self, w, x, y):
        m = y * (x @ w)
        return np.mean(np.log(1+np.exp(-m)))
    def grad(self,x,y,w):
        m = y * (x @ w)
        sigmoid = 1/(1+np.exp(-m))
        return - np.mean(y.reshape(-1,1)*x*sigmoid.reshape(-1,1))

class LinearClassification:
    def __init__(self, lr=0.01, max_iter=100):
        self.lr = lr
        self.max_iter = max_iter
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

X = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])
y = np.array([1, 1, -1, -1])
model = LinearClassification(lr=0.1, max_iter=1000)
X_train = np.insert(X, 0, 1, axis=1)  # Добавляем bias
w_init = np.random.normal(size=X_train.shape[1])
loss = LogisticLoss()

# Проверяем, что потери уменьшаются
losses = []
weights = w_init
for _ in range(100):
    gradient = loss.grad(X_train, y, weights)
    weights -= 0.1 * gradient
    losses.append(loss(weights, X_train, y))
assert all(losses[i] >= losses[i+1] for i in range(len(losses)-1)), "Ошибка: Потери не уменьшаются"
    
