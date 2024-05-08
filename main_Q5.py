import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from LinearRegression import LinearRegression
import matplotlib.pyplot as plt


if __name__ == "__main__":
    with open("..\Students_on_Mars.csv") as file:
        df = pd.read_csv(file)

    X = df.drop('y', axis=1)  # Drop the diagnosis column from features
    y = df['y']
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle the data frame
    X_train, X_test, y_train, y_test = LinearRegression.train_test_split(X, y)  # split the dataset
    result = []

    for i in range(1, 6):  # Iterate over polynomial degrees from 1 to 5
        linear_model = LinearRegression()
        poly = PolynomialFeatures(i, include_bias=False)

        X_train_poly = poly.fit_transform(X_train)  # Transform the training data to include polynomial features
        linear_model.fit(X_train_poly, y_train)

        X_test_poly = poly.fit_transform(X_test)  # Transform the test data to include polynomial features

        train_score = linear_model.score(X_train_poly, y_train)
        test_score = linear_model.score(X_test_poly, y_test)

        result.append([i, (1 - test_score) * 100, (1 - train_score) * 100])  # Append the results to the result list as error percentage

    df = pd.DataFrame(result, columns=['poly', 'test error', 'train error'])
    
    # Plotting
    plt.bar(df['poly'] - 0.2, df['test error'], width=0.4, label='Test Error')
    plt.bar(df['poly'] + 0.2, df['train error'], width=0.4, label='Train Error')

    # Adding labels and title
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Error Percentage')
    plt.xticks(df['poly'])
    plt.legend()

    # Displaying the plot
    plt.show()


