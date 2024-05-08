import numpy as np

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
        if add_bias:  # if we already add bias in the other function we don't need to add another '1' column
            ones_column = np.ones((X.shape[0], 1), dtype=X.dtype)
            X = np.hstack((ones_column, X))
        results = np.array(X@self.weights_)
        return results

    def score(self, X, y):
        u = np.sum(((y - self.predict(X))**2))
        v = np.sum((y-np.mean(y))**2)
        return 1-(u / v)
    @staticmethod
    def train_test_split(X, y, train_size=0.8):
        train_part = int(X.shape[0] * train_size)

        X_train = X[:train_part].values
        X_test = X[train_part:].values
        y_train = y[:train_part].values
        y_test = y[train_part:].values
        return X_train, X_test, y_train, y_test