import json
from LinearRegression import LinearRegression
import numpy as np
import pandas as pd

if __name__ == "__main__":
    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)
    df = pd.read_csv(config['data_file'])
    X = df.drop('y', axis=1)  # Drop the "y" column from features
    y = df['y']

    linear_regression = LinearRegression()
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle the data frame
    X_train, X_test, y_train, y_test = LinearRegression.train_test_split(X, y)  # split the dataset

    linear_regression.fit(X_train, y_train)
    print("The weights are: ", linear_regression.weights_[1:], "the b is:", linear_regression.weights_[0])
    print("The score test is: ", linear_regression.score(X_test, y_test))
