import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from LinearRegression import LinearRegression
import json

if __name__ == "__main__":
    with open('config_q5.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)
    df = pd.read_csv(config['data_file'])

    X = df.drop('y', axis=1)  # Drop the diagnosis column from features
    y = df['y']

    X_train, X_test, y_train, y_test = LinearRegression.train_test_split(X, y)  # split the dataset
    result = []
    for i in range(5):
        linear_model = LinearRegression()
        poly = PolynomialFeatures(i)
        X_train_poly = poly.fit_transform(X_train)
        y_train_poly = poly.fit_transform([y_train])
        linear_model.fit(X_train_poly, y_train_poly)
        score = linear_model.score(X_test, y_test)
        result.append([i, score])
    df = pd.DataFrame(result)
    df.plot(kind='scatter', x='x', y='y')