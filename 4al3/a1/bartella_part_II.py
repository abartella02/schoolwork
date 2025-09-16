import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import random

np.random.seed(42)

data = pd.read_csv("training_data.csv")  # read data
data = data.drop(data.columns[0], axis=1)  # drop gender column

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
for row in data.iterrows():
    if (
        all(p in row[1] for p in params) and "Rings" in row[1]
    ):  # assert target data is in row
        for param in params:
            xs[param].append(row[1][param])  # append x values
        y.append(row[1]["Rings"] + 1.5)  # add 1.5 to get age from rings


class TrainTestSet:
    """Struct to contain test and train sets for X and Y"""

    def __init__(self, X_test, X_train, Y_test, Y_train):
        self.X_test = X_test
        self.X_train = X_train
        self.Y_test = Y_test
        self.Y_train = Y_train


class PolynomialRegression:
    """Modified polynomial regression class from tutorial"""

    def __init__(
        self,
        inputs: list[np.array],
        output: list,
        train_set_ratio: float,  # percentage of the set to be used for training
        degree: int = 2,
        k_fold: int = 5,
    ):
        """Multiple x's in, one y out"""
        # VALIDATION OF PARAMETERS
        if inputs and output:  # assert inputs and outputs are not null
            # assert inputs and outputs have the same number of datapoints
            if len(inputs[0]) != len(output):
                raise ValueError(
                    "inputs and outputs must have the same number of datapoints"
                )
        else:
            raise ValueError("inputs and output must not be null")
        if k_fold < 2 or k_fold >= len(output):
            # Assert at least 2 k folds are made
            raise ValueError(
                "k_fold must be an integer greater than 1 and less than the number of datapoints."
            )
        if train_set_ratio >= 1:
            # assert training set cannot be 100% of the dataset
            raise ValueError("training set ratio must be less than 1 (100%)")
        if degree < 1:
            raise ValueError("degree must be >= 1")

        # ASSIGN INPUTS
        self.inputs = inputs  # X data
        self.output = (
            output if isinstance(output, np.ndarray) else np.array(output)
        )  # target
        self.train_set_len = int(
            round(train_set_ratio * len(self.output), 0)
        )  # training set length
        self.degree = degree  # polynomial degree
        self.k_fold = k_fold  # number of k folds to make
        self.Y_norm = []  # normalized target values
        self.X_poly = []  # normalized x values in polynomial form
        self.X_k = []  # k folds of X
        self.Y_k = []  # k folds of Y

    def normalize(self):
        """Normalize x and y data"""
        xs_norm = []
        for inp in self.inputs:
            hmean = np.mean(inp)
            hstd = np.std(inp)
            xs_norm.append((inp - hmean) / hstd)

        gmean = np.mean(self.output)
        gstd = np.std(self.output)
        y_norm = (self.output - gmean) / gstd
        return xs_norm, y_norm

    def reshape_x(self, _X):
        """
        Reshape list of separate x's into
        [
             [1, x1, x1^2, x1^3, x2, x2^2, x2^3, ... , xn, xn^2, xn^3],
             [1, x1, x1^2, x1^3, x2, x2^2, x2^3, ... , xn, xn^2, xn^3],
             ...
             [1, x1, x1^2, x1^3, x2, x2^2, x2^3, ... , xn, xn^2, xn^3]
        ]
        where n is the number of parameters (length, width, etc)
        and each row is a data point.
        Similar to numpy.column_stack()
        """
        xs_tran = np.array(_X).T
        _mat = []
        for x in xs_tran:
            _row = [1]
            for x_i in x:
                for d in range(1, self.degree + 1):
                    _row.append(x_i**d)
            _mat.append(_row)
        return np.array(_mat)

    def preprocess(self):
        """Normalize data, reshape x into polynomial form. Make k-folds."""
        _X, Y = self.normalize()
        X = self.reshape_x(_X)
        self.X_poly, self.Y_norm = X, Y

        # split into k-fold segments
        assert len(X) == len(Y)
        split = len(X) // self.k_fold
        self.X_k = [
            X[i * split : min((i + 1) * split, len(X))] for i in range(self.k_fold)
        ]
        self.Y_k = [
            Y[i * split : min((i + 1) * split, len(X))] for i in range(self.k_fold)
        ]

    def select_fold(self, k: int):
        """
        Slice X and return the desired fold along with the remaining X values
        :param k: desired fold number
        :return: Tuple containing X[k], X[ all else but k ]
        """
        Xk_1 = list(self.X_k[:k][0]) if len(self.X_k[:k]) else []  # selected k fold
        Xk_2 = (
            list(self.X_k[k + 1 :][0]) if len(self.X_k[k + 1 :]) else []
        )  # all other k folds
        Yk_1 = list(self.Y_k[:k][0]) if len(self.Y_k[:k]) else []  # selected k fold
        Yk_2 = (
            list(self.Y_k[k + 1 :][0]) if len(self.Y_k[k + 1 :]) else []
        )  # all other k folds

        # combine all other k folds
        X_not_k = np.array(Xk_1 + Xk_2)
        Y_not_k = np.array(Yk_1 + Yk_2)

        assert len(X_not_k) and len(Y_not_k)

        return TrainTestSet(self.X_k[k], X_not_k, self.Y_k[k], Y_not_k)

    def split_train_test(self):
        """Split data into training and test sets. Not k folding!"""
        return TrainTestSet(
            self.X_poly[: self.train_set_len],
            self.X_poly[self.train_set_len :],
            self.Y_norm[: self.train_set_len],
            self.Y_norm[self.train_set_len :],
        )

    def separate_x(self, X):
        """Isolate x^1 terms from polynomial X, for graphing"""
        XX_ = []
        for i in range(1, len(X.T) // self.degree + 1):
            XX_.append(X.T[self.degree * i - 1])
        return XX_

    def train(self, _X, Y):
        """Train using OLS"""
        return np.linalg.inv(_X.T.dot(_X)).dot(_X.T).dot(Y)

    def predict(self, X, beta):
        """Predict using beta"""
        beta = np.array([np.array([b]) for b in beta])
        return np.sum(X.dot(beta), axis=1)


# INIT POLYNOMIAL REGRESSION
poly_reg = PolynomialRegression(list(xs.values()), y, train_set_ratio=0.8)
poly_reg.preprocess()
t_data = poly_reg.split_train_test()

# XX = [x1, x2, x3, x4, ... xn] for plotting
XX = poly_reg.separate_x(t_data.X_test)
beta = poly_reg.train(t_data.X_train, t_data.Y_train)  # OLS
print(f"beta_prime: {beta}")
Y_predict = poly_reg.predict(t_data.X_test, beta)  # Predict Y from test set

# CV/MSE CALCULATION
MSEs = []
for i in range(poly_reg.k_fold):
    folds = poly_reg.select_fold(i)  # get desired fold
    n = len(folds.Y_test)
    MSEs.append(1 / n * sum((folds.Y_test[j] - Y_predict[j]) ** 2 for j in range(n)))


fig, axs = plt.subplots(3, 3, constrained_layout=True)
fig.set_size_inches((12.5, 6))
axs_flat = [j for sub in axs for j in sub]  # flatten matrix of plots, for iteration
# iterate through subplots, until all params exhausted
for i, ax in enumerate(axs_flat[: len(params)]):
    ax.scatter(
        XX[i], t_data.Y_test, color="g", label="Actual", alpha=0.5
    )  # Plot test data
    ax.set_ylabel("Age")  # set labels
    ax.set_xlabel(params[i])
    ax.set_title(f"Abalone age vs {params[i]}")
    CV = np.average(MSEs)  # compute CV, average of all MSEs

    # Add values to subplots
    ax.scatter(
        XX[i], Y_predict, label="Predicted", color="b", alpha=0.5
    )  # Plot predicted data
    ax.legend()
    ax.annotate(
        "MSE= %.6f" % round(CV, 6),
        xy=(1, 0),
        xycoords="axes fraction",
        xytext=(-20, 20),
        textcoords="offset pixels",
        horizontalalignment="right",
        verticalalignment="bottom",
    )  # add annotation for MSE

plt.show()
