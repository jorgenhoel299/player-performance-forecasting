import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionOwn:

    def __init__(self, lr=0.001, n_iters=5000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features)
        self.bias = 0
        dw_history = []
        db_history = []

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            dw_history.append(np.linalg.norm(dw))  # Use np.linalg.norm to get the magnitude
            db_history.append(np.abs(db))           # Use np.abs to get the absolute value

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

        # # Plot the evolution of gradients over iterations
        # plt.plot(range(1, self.n_iters + 1), dw_history, label='Gradient for Weights')
        # plt.plot(range(1, self.n_iters + 1), db_history, label='Gradient for Bias')
        # plt.xlabel('Iterations')
        # plt.ylabel('Gradient Value')
        # plt.legend()
        # plt.title('Evolution of Gradients Over Iterations')
        # plt.show()

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
