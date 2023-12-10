import numpy as np


import numpy as np

class LSM:

    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        # Add a column of ones to X for the intercept term
        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))

        # Calculate coefficients using the normal equation
        self.coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ (X_with_intercept.T @ y)

    def predict(self, X):
        # Add a column of ones to X for the intercept term
        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))

        # Predict using the coefficients
        y_pred = X_with_intercept @ self.coefficients
        return y_pred
