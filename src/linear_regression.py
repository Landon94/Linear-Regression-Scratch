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
        self.w = None
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
    
    def _compute_gradient(self, X, y):
        m,n = X.shape

        predictions = X.dot(self.w) + self.b        #shape(m,)
        error = predictions - y                     #shape(m,)

        dj_dw = X.T.dot(error)                      #shape(n,m) * shape(m,)
        dj_db = np.sum(error) / m

        return dj_dw, dj_db

    def _intilize_parameters(self, n_features):
        self.b = 0
        self.w = np.random.randn(n_features) * 0.01
    
    def fit(self, X, y):
        self.costs = [] #array to store cost history
        self._intilize_parameters(X.shape[1]) #intialize w and b
        
        for i in range(self.iterations):

            #Calculate gradient
            dj_dw, dj_db = self._compute_gradient(X, y)

            #Upadte w and b 
            self.w = self.w - self.alpha * dj_dw 
            self.b = self.b - self.alpha * dj_db

            #Save cost if < 100000 iterations
            if self.iterations < 100000:
                cost = self._compute_cost(X,y)
                self.costs.append(cost)

            #Print cost and iteration every 100 iterations
            if i % 100 == 0:
                print(f'Iteration: {i}, Cost: {cost:.4f}')

    def predict(self, x):
        return x.dot(self.W) + self.b   #shape(m,n) * shape(n,) + b


def main():
    X_train, X_test, y_train, y_test = load_data()


    X_train_normal = X_train.apply(z_score_normalization)
    Model=LinearRegression(.00027, 10000)
    Model.fit(X_train_normal, y_train)

    # print(Model.w)
    plot_features_vs_performance(X_train, y_train)
    # Model.w = np.zeros(X_train.shape[1]) 
    # Model.b = 0
    # print(Model.costs)
    plot_cost_vs_iteration(Model.costs)
    # print(Model._compute_cost(X_train, y_train))
    pass

if __name__ == "__main__":
    main()