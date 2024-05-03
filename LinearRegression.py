import numpy as np
import sympy as sp

class LinearRegression:
    def __init__(self):
        self.weights_ = None  # weights[0] represent the 'b' and [1:] represent the weights vector.

    def fit(self, X, y):
        ones_column = np.ones((X.shape[0], 1), dtype=X.dtype)  # adding a '1' column to represent the 'b' in the module.
        X = np.hstack((ones_column, X))
        num_of_cols = X.shape[1]
        X_t = np.transpose(X)
        X_tX= X_t@X
        determinant = np.linalg.det(X_tX)
        if(determinant == 0 ):
            print("cant find the optimum,X determinant is 0")
            exit(1)
        X_tX_inv = np.linalg.inv(X_tX)
        self.weights_ = X_tX_inv@X_t@y

    def predict(self, X, add_bias=True):
        if add_bias:  # if we already add bias in the fit function we don't need to add another '1' column
            ones_column = np.ones((X.shape[0], 1), dtype=X.dtype)
            X = np.hstack((ones_column, X))

        results = np.array(np.transpose(X@self.weights_))
        return results

