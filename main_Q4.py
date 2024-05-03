import pandas as pd
from LinearRegression import LinearRegression
import json
from sklearn.datasets import fetch_california_housing

if __name__ == "__main__":
    housing = fetch_california_housing(as_frame=True)
    linear_regression = LinearRegression()
    data_train, data_test, target_train, target_test = LinearRegression.train_test_split(housing.data, housing.target)
    linear_regression.fit(data_train, target_train)
    score = linear_regression.score(data_test, target_test)
    print(score)
    print("the weights are: ", linear_regression.weights_[1:], "the b is:", linear_regression.weights_[0])