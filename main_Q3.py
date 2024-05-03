import json

from LinearRegression import LinearRegression
import numpy as np
import pandas as pd

if __name__ == "__main__":
    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)
    df = pd.read_csv(config['data_file'])

    X = df.drop('y', axis=1)  # Drop the diagnosis column from features
    y = df['y']
    linear_regression = LinearRegression()

    X_train, X_test, y_train, y_test = LinearRegression.train_test_split(X, y)  # split the dataset
    #linear_regression.fit(X_train, y_train)
    linear_regression.fit(X.values, y.values)
    #y_pred = linear_regression.predict(X_test)
    print("the weights are: ", linear_regression.weights_[1:], "the b is:", linear_regression.weights_[0])
    print(linear_regression.score(X.values, y.values))