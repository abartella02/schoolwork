import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import random

np.random.seed(42)

data = pd.read_csv("gdp-vs-happiness.csv")

by_year = (data[data["Year"] == 2018]).drop(
    columns=["Continent", "Population (historical estimates)", "Code"]
)
df = by_year[
    (by_year["Cantril ladder score"].notna())
    & (by_year["GDP per capita, PPP (constant 2017 international $)"]).notna()
]

happiness = []
gdp = []
for row in df.iterrows():
    if row[1]["Cantril ladder score"] > 4.5:
        happiness.append(row[1]["Cantril ladder score"])
        gdp.append(row[1]["GDP per capita, PPP (constant 2017 international $)"])


class LinearRegression:  # From tutorial example
    def __init__(self, x_: list, y_: list) -> None:
        self.input = np.array(x_)
        self.target = np.array(y_)

    def preprocess(self):
        hmean = np.mean(self.input)
        hstd = np.std(self.input)
        x_train = (self.input - hmean) / hstd
        X = np.column_stack((np.ones(len(x_train)), x_train))
        gmean = np.mean(self.target)
        gstd = np.std(self.target)
        y_train = (self.target - gmean) / gstd
        Y = (np.column_stack(y_train)).T
        return X, Y

    def train_ols(self, X, Y):
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    def predict(self, X_test, beta):
        Y_hat = X_test * beta.T
        return np.sum(Y_hat, axis=1)


def gradient_descent(
    _X: np.ndarray, _Y: np.ndarray, _n: int, _alpha: float, _epoch: int
):
    _beta = np.random.randn(_X.shape[1], 1)
    for _ in range(_epoch):
        gradients = 2 / _n * _X.T.dot(_X.dot(_beta) - _Y)
        _beta -= _alpha * gradients
    print(f"Epochs = {_epoch}\nLearning Rate = {_alpha}\nBeta prime = {_beta}")
    return _beta


lr = LinearRegression(gdp, happiness)  # init linear regression

X, Y = lr.preprocess()  # normalize values and reshape
X_ = X[..., 1].ravel()  # flatten X for graphing

axs = []
for _ in range(2):
    fig, ax = plt.subplots()
    axs.append(ax)
    fig.set_size_inches((10, 5))
    ax.scatter(X_, Y)  # plot actual values
    ax.set_ylabel("Happiness")
    ax.set_xlabel("GDP per capita")
    ax.set_title("Cantril Ladder Score vs GDP per capita of countries (2018)")


# Output one - regression lines from gd
n: int = len(X)
"""Number of observations"""

alphas: list[float] = [0.7, 0.9]
"""Learning rate"""

epochs: list[int] = [2, 5, 7]
"""Number of iterations"""

beta = [[0] for _ in range(X.shape[1])]  # init beta
for alpha in alphas:
    for epoch in epochs:
        beta = gradient_descent(
            X, Y, n, alpha, epoch
        )  # perform gradient descent operation
        Y_predict = lr.predict(X, beta)  # get predicted y values
        # plot predicted y values vs actual x vals
        axs[0].plot(X_, Y_predict, label=f"[{epoch}, {alpha}]")
        # calculate MSE
        MSE = 1 / n * sum((Y[j][0] - Y_predict[j]) ** 2 for j in range(n))
        print(f"MSE: {MSE}\n")

Y_predict_ols = lr.predict(X, lr.train_ols(X, Y))  # get ols results

# Output two - best gd result
best = (0.7, 10)
best_beta = gradient_descent(X, Y, n, *best)  # get best beta
Y_predict_best = lr.predict(X, np.array(best_beta))  # predict using best beta
MSE = 1 / n * sum((Y[j][0] - Y_predict_best[j]) ** 2 for j in range(n))  # calculate MSE
print(f"MSE: {MSE}")

# plot predicted via ols
axs[1].plot(X_, Y_predict_ols, label="OLS result", color="k", linewidth=4)
# plot predicted via best beta
axs[1].plot(X_, Y_predict_best, label=f"Best via GD\n{best}", color="r", linewidth=0.5)

axs[0].legend(title="[Epoch, Learning rate]")
axs[1].legend()
plt.show()
