import numpy as np


class LinearRegression:

    def __init__(self, lr = 0.001, n_iters=5000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
    # Get the number of samples and features
        n_samples, n_features = X.shape

        # Initialize weights with small random values
        #self.weights = np.random.randn(1)
        self.weights = np.array([0.5])
        self.bias = 0

        # Iterate through training iterations
        for _ in range(self.n_iters):
            # Make predictions using current weights and bias
            y_pred = np.dot(X, self.weights) + self.bias

            # Calculate gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            # Update weights and bias using gradients and learning rate
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db



    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
