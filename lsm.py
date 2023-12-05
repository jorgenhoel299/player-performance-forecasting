import numpy as np


class LSM:

    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, X, y):
    # Get the number of samples and features
        X_mean = np.mean(X)
        y_mean = np.mean(y)

        num = 0
        den = 0
        for i in range(len(X)):
            num += (X[i] - X_mean)*(y[i] - y_mean)
            den += (X[i] - X_mean)**2
        self.slope = num/den
        self.intercept = y_mean - (self.slope * X_mean)

    def predict(self, X):
        y_pred = np.dot(X, self.slope) + self.intercept
        return y_pred
