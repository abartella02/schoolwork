import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle


class svm_():
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
        self._loss = None

    def pre_process(self, ):

        # using StandardScaler to normalize the input
        scaler = StandardScaler()
        X_ = scaler.fit_transform(self.input)

        Y_ = self.target

        return X_, Y_

        # the function return gradient for 1 instance -

    # stochastic gradient decent
    def compute_gradient(self, X, Y):
        # organize the array as vector
        X_ = np.array([X])

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
        """calculate hinge loss"""
        # hinge loss implementation- start
        # hinge_loss=max(0,1−y⋅(w⋅x+b))

        loss = np.mean(
            np.maximum(
                0,
                1 - Y * np.dot(X, self.weights)

            )
        )
        # hinge loss implementation - end

        return loss

    def stochastic_gradient_descent(self, X, Y):

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        samples = 0

        interval = self.epoch // 10

        tolerance = 0.001
        total_patience = 10
        patience = total_patience

        prev_loss = np.inf
        test_losses = []
        epoch_x = []

        stop_loss = (0, 0)
        stop_flag = False
        # execute the stochastic gradient descent function for defined epochs
        for epoch in range(self.epoch):

            # shuffle to prevent repeating update cycles
            train_features, train_output = shuffle(X_train, y_train)
            test_features, test_output = shuffle(X_test, y_test)

            # update the weights by doing gradient descent on the train set
            for i, feature in enumerate(train_features):
                gradient = self.compute_gradient(feature, train_output[i])
                self.weights = self.weights - (self.learning_rate * gradient)

            # calculate the loss on the test set using the new weights
            loss = self.compute_loss(test_features, test_output)

            if abs(prev_loss - loss) < tolerance:
                patience -= 1

            if patience == 0 and not stop_flag:
                print(f"Stopping early at {epoch}")
                stop_flag = True

            if stop_flag:
                self.learning_rate = self.learning_rate / 2

            # check for convergence - end

            # below code will be required for Part 3

            # Part 3

            if epoch % interval == 0:
                epoch_x.append(epoch)
                test_losses.append(loss)
            prev_loss = loss
            samples += 1

        print("Training ended...")
        print(f"weights are: {self.weights}")

        # below code will be required for Part 3
        print("The minimum number of samples used are:", samples)
        fig, ax = plt.subplots()
        ax.plot(epoch_x, test_losses, '-o')
        # ax.plot(*stop_loss, '-o', color='k')
        self._loss = test_losses[-1]
        return test_losses[-1]

    def mini_batch_gradient_descent(self, X, Y, batch_size):

        # mini batch gradient decent implementation - start

        # Part 2

        # mini batch gradient decent implementation - end

        print("Training ended...")
        print("weights are: {}".format(self.weights))

    def sampling_strategy(self, X, Y):
        x = X[0]
        y = Y[0]
        # implementation of sampling strategy - start

        # Part 3

        # implementation of sampling strategy - start
        return x, y

    def predict(self, X_test, Y_test):

        # compute predictions on test set
        predicted_values = [np.sign(np.dot(X_test[i], self.weights)) for i in range(X_test.shape[0])]

        # compute accuracy
        accuracy = accuracy_score(Y_test, predicted_values)
        print(f"Accuracy on test dataset: {accuracy}")

        # compute precision - start
        # Part 2
        # compute precision - end

        # compute recall - start
        # Part 2
        # compute recall - end
        return predicted_values, accuracy


def part_1(X_train, y_train, c, lr, ep):
    # model parameters - try different ones
    # C = 0.1
    # learning_rate = 0.4
    # epoch = 4000
    C = c
    learning_rate = lr
    epoch = ep

    # intantiate the support vector machine class above
    my_svm = svm_(learning_rate=learning_rate, epoch=epoch, C_value=C, X=X_train, Y=y_train)

    # pre preocess data
    X, Y = my_svm.pre_process()

    # train model
    my_svm.stochastic_gradient_descent(X, Y)

    return my_svm


def part_2(X_train, y_train):
    # model parameters - try different ones
    C = 0.001
    learning_rate = 0.001
    epoch = 5000

    # intantiate the support vector machine class above
    my_svm = svm_(learning_rate=learning_rate, epoch=epoch, C_value=C, X=X_train, Y=y_train)

    # pre preocess data

    # select samples for training

    # train model

    return my_svm


def part_3(X_train, y_train):
    # model parameters - try different ones
    C = 0.001
    learning_rate = 0.001
    epoch = 5000

    # intantiate the support vector machine class above
    my_svm = svm_(learning_rate=learning_rate, epoch=epoch, C_value=C, X=X_train, Y=y_train)

    # pre preocess data

    # select samples for training

    # train model

    return my_svm


if __name__ == "__main__":
    # Load datapoints in a pandas dataframe
    print("Loading dataset...")
    data = pd.read_csv('data.csv')

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
    category_dict = {'B': -1.0, 'M': 1.0}
    # transpose to column vector
    Y = np.array([(data.loc[:, 'diagnosis']).to_numpy()]).T
    Y_target = np.vectorize(category_dict.get)(Y)

    # split data into train and test set using sklearn feature set
    print("splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_features, Y_target, test_size=0.2, random_state=42)

    # my_svm = part_1(X_train, y_train)

    # my_svm = part_2(X_train, y_train)

    # my_svm = part_3(X_train, y_train)

    C = [x/100 for x in range(1, 200, 5)]
    learning_rate = [x/100 for x in range(1, 200, 4)]
    epoch = range(200, 5000, 300)
    results = []

    for c in C:
        for lr in learning_rate:
            for ep in epoch:
                my_svm = part_1(X_train, y_train, c, lr, ep)

                scaler = StandardScaler()
                X_Test_Norm = scaler.fit_transform(X_test)

                # testing the model
                print("Testing model accuracy...")
                _, accuracy = my_svm.predict(X_Test_Norm, y_test)
                results.append((c, lr, ep, my_svm._loss, accuracy))

    res = results.copy()
    for i in range(11):
        best_result = max(res, key=lambda x: (x[-1]-x[-2]))
        print("Best results based on (accuracy - loss)")
        print(f"{i}  -  c: {best_result[0]}, lr: {best_result[1]}, ep: {best_result[2]}, loss: {best_result[3]}, accuracy: {best_result[4]}")
        res.remove(best_result)

    a = 1
    #plt.show()
