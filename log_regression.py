import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegressionTest:
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            linear_predictions = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_predictions)

            gradient = (1 / n_samples) * np.dot(X.T, (predictions - y))
            gradient_bias = (1 / n_samples) * np.sum(predictions - y)

            self.weights = self.weights - self.learning_rate * gradient
            self.bias = self.bias - self.learning_rate * gradient_bias

    def predict(self, X):
        linear_predictions = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_predictions)
        #class_pred = [0 if y <= 0.67 else 1 for y in y_pred]
        class_pred = y_pred
        return class_pred
