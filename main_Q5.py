import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from LinearRegression import LinearRegression
import json
import matplotlib.pyplot as plt


if __name__ == "__main__":
    with open('config_q5.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)
    df = pd.read_csv(config['data_file'])

    X = df.drop('y', axis=1)  # Drop the diagnosis column from features
    y = df['y']

    X_train, X_test, y_train, y_test = LinearRegression.train_test_split(X, y)  # split the dataset
    result = []

    for i in range(1, 4):
        linear_model = LinearRegression()
        poly = PolynomialFeatures(i, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)

        linear_model.fit(X_train_poly, y_train)
        X_test_poly = poly.fit_transform(X_test)
        train_score = linear_model.score(X_train_poly, y_train)
        test_score = linear_model.score(X_test_poly, y_test)
        result.append([i, (1 - test_score) * 100, (1 - train_score) * 100])

    df = pd.DataFrame(result, columns=['poly', 'test error', 'train error'])
    # Plotting
    fig, ax = plt.subplots()
    bar_width = 0.35
    bar_poly = df['poly']
    bar_test_error = df['test error']
    bar_train_error = df['train error']
    index = range(len(bar_poly))

    bars1 = ax.bar(index, bar_test_error, bar_width, label='Test Error')
    bars2 = ax.bar([i + bar_width for i in index], bar_train_error, bar_width, label='Train Error')

    # Adding labels and title
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Error Percentage')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(bar_poly)
    ax.legend()

    # Displaying the plot
    plt.show()


