import numpy as np
import pandas as pd
from utils import *

def mean_normalization(X) -> pd.Series:
    return (X - X.mean()) / (X.max() - X.min())


def z_score_normalization(X) -> pd.Series:
    return (X - X.mean()) / X.std()


class LinearRegression():

    def __init__(self, alpha, iterations):
        self.alpha = alpha
        self.iterations = iterations
        self.W = None
        self.b = None

    def _compute_cost(self, X, y):
        """
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)) : target values
        w (ndarray (n,)) : model parameters  
        b (scalar)       : model parameter
        """
        m = X.shape[0]
        cost = 0.0
        predictions = X.dot(self.w) + self.b
        cost = np.sum((predictions - y) ** 2)
        return cost / (2 * m)
        

    def predict(self, x):
        return x.dot(self.W) + self.b


def main():
    X_train, X_test, y_train, y_test = load_data()
    Model=LinearRegression(.01, 100)
    Model.w = np.zeros(X_train.shape[1]) 
    Model.b = 0

    print(Model._compute_cost(X_train, y_train))

if __name__ == "__main__":
    main()