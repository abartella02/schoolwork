import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle


class svm_:
    def __init__(self, learning_rate, epoch, C_value, X, Y):

        # initialize the variables
        self.input = X
        self.target = Y
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.C = C_value

        # initialize the weight matrix based on number of features
        # bias and weights are merged together as one matrix
        # you should try random initialization

        self.weights = np.zeros(X.shape[1])

    def pre_process(
        self,
    ):

        # using StandardScaler to normalize the input
        scaler = StandardScaler()
        X_ = scaler.fit_transform(self.input)

        Y_ = self.target

        return X_, Y_

    # the function return gradient for 1 instance -
    # stochastic gradient decent
    def compute_gradient(self, X, Y):
        """The function return gradient for 1 instance"""

        X_ = np.array([X])  # organize the array as vector

        # hinge loss
        hinge_distance = 1 - (Y * np.dot(X_, self.weights))

        total_distance = np.zeros(len(self.weights))
        # hinge loss is not defined at 0
        # is distance equal to 0
        if max(0, hinge_distance[0]) == 0:
            total_distance += self.weights
        else:
            total_distance += self.weights - (self.C * Y[0] * X_[0])

        return total_distance

    def compute_loss(self, X, Y):
        """calculate average hinge loss"""
        # hinge_loss=max(0,1−y⋅(w⋅x+b))
        loss = 0
        for i in range(len(X)):
            loss += np.maximum(0, 1 - Y[i] * np.dot(self.weights, X[i]))
        return loss / len(X)

    def stochastic_gradient_descent(self, X, Y):
        # Split the training set to create a mini-validation set, to evaluate loss
        X_train, test_features, y_train, test_output = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        interval = self.epoch // 10  # interval to graph loss

        tolerance = 0.001  # tolerance for stopping
        total_patience = (
            5  # patience: how many times the tolerance can be violated before stopping
        )
        patience = total_patience  # start at full patience

        prev_loss = np.inf  # init prev epoch's loss
        test_losses = []  # lists to contain losses which will be graphed
        train_losses = []
        epoch_x = []
        early_stop = (0, 0)  # init early stop point

        stop_flag = False  # init stop flag
        # execute the stochastic gradient descent function for defined epochs
        for epoch in range(self.epoch):

            # shuffle to prevent repeating update cycles
            train_features, train_output = shuffle(
                X_train, y_train
            )  # shuffle training set

            # update the weights by doing gradient descent on the train set
            for feature, output in zip(train_features, train_output):
                gradient = self.compute_gradient(feature, output)
                self.weights = self.weights - (self.learning_rate * gradient)

            # calculate the loss on the test set using the new weights
            loss = self.compute_loss(test_features, test_output)

            # check for tolerance violation
            if abs(prev_loss - loss) < tolerance:
                patience -= 1  # decrement patience if tolerance is violated
            prev_loss = loss  # set previous epoch's loss to current loss

            if epoch % interval == 0:  # save a datapoint at each 1/10th epoch
                epoch_x.append(epoch)
                test_losses.append(loss)
                train_losses.append(
                    self.compute_loss(train_features, train_output)
                )  # compute loss on training set

            if (
                patience == 0 and not stop_flag
            ):  # if patience runs out, convergence detected
                print(f"Stopping early at {epoch}")
                early_stop = (epoch, loss)  # note early stop point
                stop_flag = (
                    True  # stop flag to not print/save early stop at further epochs
                )

        # print("Training ended...")
        # print(f"weights are: {self.weights}")

        return epoch_x, test_losses, train_losses, early_stop

    def mini_batch_gradient_descent(self, X, Y, batch_size, test_size=0.2):
        # Split the training set to create a mini-validation set, to evaluate loss
        X_train, test_features, y_train, test_output = train_test_split(
            X, Y, test_size=test_size, random_state=42
        )

        interval = self.epoch // 10  # interval to graph loss

        vali_losses = []  # lists to contain losses which will be graphed
        train_losses = []
        epoch_x = []

        # execute the stochastic gradient descent function for defined epochs
        for epoch in range(self.epoch):

            # shuffle to prevent repeating update cycles
            train_features, train_output = shuffle(X_train, y_train)

            # update the weights by doing gradient descent on the train set
            for idx in range(0, len(train_features), batch_size):
                # create batches by slicing training set
                batch_features = train_features[idx : idx + batch_size]
                batch_output = train_output[idx : idx + batch_size]

                i = 0
                gradients = 0  # init gradient sum
                for feature, output in zip(batch_features, batch_output):
                    gradients += self.compute_gradient(feature, output)
                    i += 1
                # recalculate weights with average gradients
                self.weights = self.weights - (self.learning_rate * (gradients / i))

            if epoch % interval == 0:  # save a datapoint at each 1/10th epoch
                epoch_x.append(epoch)
                vali_losses.append(
                    self.compute_loss(test_features, test_output)
                )  # compute loss on test set
                train_losses.append(
                    self.compute_loss(train_features, train_output)
                )  # compute loss on training set

        # print("Training ended...")
        # print(f"weights are: {self.weights}")

        return epoch_x, vali_losses, train_losses

    def active_learning(self, X, Y, threshold=0.01):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))  # init plots

        n = 10  # n samples
        samples = n  # track the total sampled used

        prev_loss = np.inf  # init previous iteration's loss
        total_patience = 5  # total patience before detecting convergence
        patience = total_patience  # set patience

        # Make subset n of full dataset
        features, output = shuffle(X, Y)  # select n samples randomly
        features_n, output_n = features[:n], output[:n]  # subset n
        features_N, output_N = features[n:], output[n:]  # N-n

        overall_losses = []  # track loss vs # of samples
        overall_samples = []

        # train on just a few labeled data (n)
        # self.mini_batch_gradient_descent(features_n, output_n, 4, test_size=0.2)
        self.stochastic_gradient_descent(features_n, output_n)

        # select additional sample and retrain
        while len(features_N) > 0:
            # choose sample based on sample with lowest loss
            choice = self.sampling_strategy(features_N, output_N)

            # Remove chosen datapoint from N
            features_N = np.vstack(
                (
                    features_N[: choice["idx"]],
                    features_N[min(choice["idx"] + 1, len(features_N)) :],
                )
            )
            output_N = np.vstack(
                (
                    output_N[: choice["idx"]],
                    output_N[min(choice["idx"] + 1, len(output_N)) :],
                )
            )

            # Append chosen datapoint to n
            features_n = np.vstack((features_n, choice["features"]))
            output_n = np.vstack((output_n, choice["label"]))

            # Train again on new set of n points
            epoch_x, vali_losses, train_losses, _ = self.stochastic_gradient_descent(
                features_n, output_n
            )
            # epoch_x, vali_losses, train_losses = self.mini_batch_gradient_descent(features_n, output_n, 10)

            curr_loss = self.compute_loss(
                features_N, output_N
            )  # compute current iteration's loss
            samples += 1

            overall_losses.append(curr_loss)
            overall_samples.append(samples)

            # plot train and validation losses for this # of samples
            ax1.plot(epoch_x, vali_losses, label=f"{samples}")
            ax2.plot(epoch_x, train_losses, label=f"{samples}")

            # if convergence is detected (same method as early stopping)
            if abs(prev_loss - curr_loss) < threshold:
                patience -= 1  # decrement patience when convergence detected
            else:
                patience = (
                    total_patience  # replenish patience if convergence not detected
                )
            prev_loss = (
                curr_loss  # set previous iteration's loss to current iterations loss
            )

            if patience == 0:  # if enough convergence detected, break
                print("Convergence reached")
                break

        ## Plotting results ##
        fontP = FontProperties()
        fontP.set_size("xx-small")  # set legend font to extra small

        # Setup validation losses graph
        ax1.set_title("Validation losses")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Hinge loss")
        ax1.legend(
            title="# samples", loc="center left", bbox_to_anchor=(1.04, 0.5), prop=fontP
        )

        # Setup training losses graph
        ax2.set_title("Training losses")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Hinge loss")
        ax2.legend(
            title="# samples", loc="center left", bbox_to_anchor=(1.04, 0.5), prop=fontP
        )

        # Plotting overall change in loss compared to number of samples
        ax3.plot(overall_samples, overall_losses)
        ax3.set_xlabel("Samples")
        ax3.set_ylabel("Hinge loss")
        ax3.set_ylim([0, 1])
        ax3.set_title("Change in validation losses vs samples")

        plt.subplots_adjust(wspace=0.7)  # for legend placement

        print(f"Samples used to achieve optimal performance: {samples}")

    def sampling_strategy(self, X, Y):
        losses = []
        # iterate through features and labels
        for i, (feat, label) in enumerate(zip(X, Y)):
            # compute loss of each individual datapoint
            losses.append(
                {
                    "idx": i,
                    "features": feat,
                    "label": label,
                    "loss": self.compute_loss([feat], [label])[0],
                }
            )

        return min(
            losses, key=lambda x: x["loss"]
        )  # find minimal loss and return associated datapoint

    def predict(self, X_test, Y_test):

        # compute predictions on test set
        predicted_values = [
            np.sign(np.dot(X_test[i], self.weights)) for i in range(X_test.shape[0])
        ]

        # compute accuracy
        accuracy = accuracy_score(Y_test, predicted_values)
        print(f"\tAccuracy on test dataset: {accuracy}")

        # compute precision - start
        precision = precision_score(Y_test, predicted_values)
        print(f"\tPrecision on test dataset: {precision}")
        # compute precision - end

        # compute recall - start
        recall = recall_score(Y_test, predicted_values)
        print(f"\tRecall on test dataset: {recall}")
        # compute recall - end
        return predicted_values, accuracy


def part_1(X_train, y_train):
    # model parameters
    C = 0.1
    learning_rate = 0.001
    epoch = 4000

    # intantiate the support vector machine class
    my_svm = svm_(
        C_value=C, learning_rate=learning_rate, epoch=epoch, X=X_train, Y=y_train
    )

    # pre preocess data
    X, Y = my_svm.pre_process()

    # train model on stochastic GD
    es_x, es_test, es_train, stop_pt = my_svm.stochastic_gradient_descent(X, Y)

    # plot the loss over epochs
    fig, ax = plt.subplots()
    ax.plot(es_x, es_test, "-o", color="b", label="Test set loss")
    ax.plot(es_x, es_train, "-o", color="g", label="Training set loss")
    ax.plot(*stop_pt, "-o", color="k", label="Early stop point")
    ax.axhline(y=stop_pt[1], color="r", linestyle="--", label="Early stop loss")
    ax.set_title("Stochastic Gradient Descent")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Hinge loss")
    ax.legend()

    return my_svm


def part_2(X_train, y_train):
    # model parameters
    C = 0.1
    learning_rate = 0.001
    epoch = 4000
    batch_size = 20

    # intantiate the support vector machine class
    my_svm = svm_(
        C_value=C, learning_rate=learning_rate, epoch=epoch, X=X_train, Y=y_train
    )

    # pre prepcess data
    X, Y = my_svm.pre_process()

    # train on mini batch GD
    batch_x, batch_test, batch_train = my_svm.mini_batch_gradient_descent(
        X, Y, batch_size
    )

    # Get data to compare with stochastic GD
    my_svm_pt1 = svm_(
        C_value=C, learning_rate=learning_rate, epoch=epoch, X=X_train, Y=y_train
    )
    _X, _Y = my_svm.pre_process()
    es_x, es_test, es_train, stop_pt = my_svm_pt1.stochastic_gradient_descent(_X, _Y)

    # Plot the loss over epochs for batch GD
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(batch_x, batch_test, "-o", color="b", label="Test set loss")
    ax1.plot(batch_x, batch_train, "-o", color="g", label="Training set loss")
    ax1.set_title("Batch Gradient Descent")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Hinge loss")
    ax1.legend()

    # Plot the loss over epochs for stochastic GD comparison
    ax2.plot(es_x, es_test, "-o", color="b", label="Test set loss")
    ax2.plot(es_x, es_train, "-o", color="g", label="Training set loss")
    ax2.plot(*stop_pt, "-o", color="k", label="Early stop point")
    ax2.axhline(y=stop_pt[1], color="r", linestyle="--", label="Early stop loss")
    ax2.set_title("Stochastic Gradient Descent")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Hinge loss")
    ax2.legend()

    return my_svm


def part_3(X_train, y_train):
    # model parameters
    C = 0.1
    learning_rate = 0.0005
    epoch = 2000

    # intantiate the support vector machine class
    my_svm = svm_(
        C_value=C, learning_rate=learning_rate, epoch=epoch, X=X_train, Y=y_train
    )

    # pre preocess data
    X, Y = my_svm.pre_process()

    # train model using active learning strategy
    my_svm.active_learning(X, Y, threshold=0.01)

    return my_svm


if __name__ == "__main__":
    # Load datapoints in a pandas dataframe
    print("Loading dataset...")
    data = pd.read_csv("data.csv")

    # drop first and last column
    data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

    # segregate inputs and targets

    # inputs
    X = data.iloc[:, 1:]

    # add column for bias
    X.insert(loc=len(X.columns), column="bias", value=1)
    X_features = X.to_numpy()

    # converting categorical variables to integers
    # - this is same as using one hot encoding from sklearn
    # benign = -1, melignant = 1
    category_dict = {"B": -1.0, "M": 1.0}
    # transpose to column vector
    Y = np.array([(data.loc[:, "diagnosis"]).to_numpy()]).T
    Y_target = np.vectorize(category_dict.get)(Y)

    # split data into train and test set using sklearn feature set
    print("splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, Y_target, test_size=0.2, random_state=42
    )

    # my_svm = part_1(X_train, y_train)

    # my_svm = part_2(X_train, y_train)

    my_svm = part_3(X_train, y_train)

    # normalize the test set separately
    scaler = StandardScaler()
    X_test_Norm = scaler.fit_transform(X_test)

    # testing the model
    print("Testing model accuracy...")
    my_svm.predict(X_test_Norm, y_test)
    plt.show()  # show plots
