import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import random

np.random.seed(42)

data = pd.read_csv("training_data.csv")
data = data.drop(data.columns[0], axis=1)  # drop gender column

split = 1300  # Split data into training set and test set
training_data = data[:split]
testing_data = data[split:]

# list of parameters of which to train the model on
params = [
    "Length",
    "Diameter",
    "Height",
    "Whole_weight",
    "Shucked_weight",
    "Viscera_weight",
    "Shell_weight",
]

# Collection of x values for each parameter
# i.e. xs["Length"] = [5 cm, 6 cm, 3 cm, etc]
xs = {param: [] for param in params}

y = []  # Age values

# Add corresponding parameter and ring values into respective structures
for row in training_data.iterrows():
    if all(p in row[1] for p in params) and "Rings" in row[1]:
        for param in params:
            xs[param].append(row[1][param])
        y.append(row[1]["Rings"] + 1.5)  # add 1.5 to get age from rings


class PolynomialRegression:
    def __init__(self, xx_: dict[str, list], y_: list, degree: int = 2) -> None:
        """
        xx_ - dict containing x values of each parameter in a respective list
        i.e. xx_["Length"] = [5 cm, 6 cm, 3 cm, etc]
        y_ - y values for the corresponding x's
        degree - degree of polynomial used (default = 2)
        """
        # convert contents of xx_ to numpy arrays
        self.inputs = [np.array(xx_[k]) for k in xx_.keys()]
        self.target = np.array(y_)  # convert y to np array
        self.degree = degree

    def preprocess(self):
        # normalize the values
        xx_train = []
        for inp in self.inputs:
            hmean = np.mean(inp)
            hstd = np.std(inp)
            xx_train.append((inp - hmean) / hstd)

        # arrange in matrix format
        # [
        #      [1, x1, x1^2, x1^3, x2, x2^2, x2^3, ... , xn, xn^2, xn^3],
        #      [1, x1, x1^2, x1^3, x2, x2^2, x2^3, ... , xn, xn^2, xn^3],
        #      ...
        #      [1, x1, x1^2, x1^3, x2, x2^2, x2^3, ... , xn, xn^2, xn^3]
        # ]
        # where n is the number of parameters (length, width, etc)
        # each row is a data point
        xx_t = np.array(xx_train).T  # get transverse of xx
        _col = []
        for x in xx_t:
            _row = [1]  # for row of 1's
            for x_i in x:
                # add one x, one x^2, one x^3, ... to degree specified
                for d in range(1, self.degree + 1):
                    _row.append(x_i**d)
            _col.append(_row)

        _X = np.array(_col)  # save as numpy array

        # normalize the values
        gmean = np.mean(self.target)
        gstd = np.std(self.target)
        y_train = (self.target - gmean) / gstd

        # arrange in matrix format
        _Y = (np.array([y_train])).T

        return _X, _Y

    def train(self, X, Y):
        # compute using OLS and return beta
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    def predict(self, x_test, _beta):
        # fitting is done in preprocessing
        # predict the y points using x points
        # x_test =
        # [
        #   [1, 904239043, 4895389359, 340880335]
        #   ...
        # ]
        # x_test.dot(beta) is column vector
        return np.sum(x_test.dot(_beta), axis=1)


poly_reg = PolynomialRegression(xs, y)
X, Y = poly_reg.preprocess()

# sort X matrix: separate into x1, x2, x3, ...
# i.e. XX_ = [x1 values, x2 values, x3 values ...]
XX_ = []
for i in range(1, len(X.T) // 2 + 1):
    XX_.append(X.T[2 * i - 1])

# Initialize plots
fig, axs = plt.subplots(3, 3, constrained_layout=True)
fig.set_size_inches((12.5, 6))
axs_flat = [j for sub in axs for j in sub]  # flatten matrix of plots, for iteration
# iterate through subplots, until all params exhausted
for i, ax in enumerate(axs_flat[: len(params)]):
    ax.scatter(XX_[i], Y, color="g", label="Actual", alpha=0.5)
    ax.set_ylabel("Age")
    ax.set_xlabel(params[i])
    ax.set_title(f"Abalone age vs {params[i]}")

# Output one - predicted values scatter from ols
beta = poly_reg.train(X, Y)  # OLS
print(f"beta_prime: {beta}")
Y_predict = poly_reg.predict(X, beta)

# Add values to subplots
for i, ax in enumerate(axs_flat[: (len(params))]):
    ax.scatter(XX_[i], Y_predict, label="Predicted", color="b", alpha=0.5)
    ax.legend()

n = len(Y)
MSE = 1/n*sum((Y[i][0]-Y_predict[i])**2 for i in range(n))

plt.show()
