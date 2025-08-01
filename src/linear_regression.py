import numpy as np
import pandas as pd
from utils import *
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def mean_normalization(X) -> pd.Series:
    return (X - X.mean()) / (X.max() - X.min())


def z_score_normalization(X) -> pd.Series:
    return (X - X.mean()) / X.std()


class LinearRegression():

    def __init__(self, alpha, iterations, regularization=True, lambda_=0.0):
        self.alpha = alpha
        self.iterations = iterations
        self.regularization = regularization
        self.lambda_ = lambda_
        self.w = None
        self.b = None

    def _compute_cost(self, X, y) -> float:
        """
        X (DataFrame (m,n)): Data, m examples with n features
        y (Series (m,)) : target values
        w (Series (n,)) : model parameters  
        b (scalar)       : model parameter
        """
        m = X.shape[0]
        cost = 0.0
        predictions = X.dot(self.w) + self.b
        cost = np.sum((predictions - y) ** 2)

        if self.regularization:
            cost += (self.lambda_ / 2) * np.sum(self.w ** 2)
        return cost / (2 * m)
    
    def _compute_gradient(self, X, y) -> float:
        m,n = X.shape

        predictions = X.dot(self.w) + self.b        #shape(m,)
        error = predictions - y                     #shape(m,)

        if self.regularization:
            dj_dw = X.T.dot(error) + (self.lambda_ / m) * self.w  #shape(n,m) * shape(m,)
        else:
            dj_dw = X.T.dot(error)
        dj_db = np.sum(error) / m

        return dj_dw, dj_db

    def _intilize_parameters(self, n_features) -> None:
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

    def predict(self, x) -> float:
        return x.dot(self.w) + self.b   #shape(m,n) * shape(n,) + b


def main():
    #Load in test Data
    X_train, X_test, y_train, y_test = load_data()

    #Normalize the data
    mean = X_train.mean()
    std = X_train.std()
    X_train_normal = X_train.apply(z_score_normalization)
    X_test_normal = (X_test - mean) / std

    #Optional: Visual features vs performance
    # plot_features_vs_performance(X_train, y_train)
    # plot_features_vs_performance(X_train_normal, y_train, norm=True)
    
    #Intialize Model and train
    learning_rate = .00027
    iterations = 25000
    lambda_ = .001
    Model=LinearRegression(learning_rate, iterations, regularization=True, lambda_=lambda_)
    Model.fit(X_train_normal, y_train)
    
    #Evaluate Model
    predictions = Model.predict(X_test_normal)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Test MSE: {mse:.2f}")
    print(f"Test R²: {r2:.2f}")

    plot_cost_vs_iteration(Model.costs)

    plot_predictions_vs_actual(y_test, predictions)


if __name__ == "__main__":
    main()