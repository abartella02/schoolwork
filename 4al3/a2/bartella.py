import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import sklearn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Fset(np.ndarray):
    """Feature set object, adds name attribute to ndarray"""

    def __new__(cls, input_array, name=""):
        obj = np.asarray(input_array).view(cls)  # create np.ndarray
        obj.name = name  # add name attr to object
        return obj


class KFold:
    """Kfold object which holds test and training sets"""

    def __init__(
        self, train_X: np.ndarray, train_Y: list, test_X: np.ndarray, test_Y: list
    ):
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y


class KFolds:
    """Create k folds from feature set"""

    def __init__(self, X: Fset, Y: list, k: int):
        def folds(_feature_set, _k: int):
            """Create folds"""
            fold_length = len(_feature_set) // _k
            return [
                _feature_set[
                    fold_length * i : min(fold_length * (i + 1), len(_feature_set))
                ]
                for i in range(_k)
            ]

        self.X_folds = folds(X, k)
        self.Y_folds = folds(Y, k)

    def get_train_test(self, fold_num: int):
        """Retrieve specified fold's train and test sets"""
        test_set_x = self.X_folds[fold_num]  # select fold to use as test set
        train_set_x = np.vstack(
            tuple(fold for j, fold in enumerate(self.X_folds) if j != fold_num)
        )  # combine all other folds to create training set

        # repeat for y values
        test_set_y = self.Y_folds[fold_num]
        train_set_y = []
        _ = [
            train_set_y := train_set_y + fold
            for j, fold in enumerate(self.Y_folds)
            if j != fold_num
        ]

        return KFold(train_set_x, train_set_y, test_set_x, test_set_y)


class SVM:
    def __init__(self, _dir: str):
        files = os.scandir(_dir)  # get all files in 'data-2010-2015/' etc
        data = {}
        for file in files:
            if (
                file.is_file()
                and file.name.endswith(".npy")
                and file.name
                not in [
                    "goes_data.npy",
                    "data_order.npy",
                ]
            ):
                data[file.name.removesuffix(".npy")] = np.load(
                    f"{_dir}/{file.name}", allow_pickle=True
                )  # Store datasets in a dictionary
        self.dir = _dir
        self.data = data
        self.data_order = np.load(f"{_dir}/data_order.npy")  # Store data order array

        self.data_arr = {}
        self.class_arr = {}
        self.labels = []

    # preprocess() function:
    #  1) normalizes the data,
    #  2) removes missing values
    #  3) assign labels to target
    def preprocess(self) -> None:
        data_norm = {}
        for key, data in self.data.items():
            norm_data = (
                StandardScaler().fit_transform(data)
                # dont normalize datasets exempt from normalization
                if key not in ["data_order", "neg_class", "pos_class"]
                else data
            )
            data_norm[key] = norm_data  # save normalized data

        # Combine positive and negative data sets
        data_stack = {}
        feature_sets = [
            ("pos_features_historical", "neg_features_historical"),
            ("pos_features_main_timechange", "neg_features_main_timechange"),
            ("pos_features_maxmin", "neg_features_maxmin"),
        ]
        for pos, neg in feature_sets:
            data_stack[pos.removeprefix("pos_")] = np.vstack(
                (data_norm[pos], data_norm[neg])
            )  # combine and store datasets

        # combine pos_class and neg_class
        class_stack = np.vstack((data_norm["pos_class"], data_norm["neg_class"]))

        # rearrange all data via data order
        data_arr = {}  # arranged data dict
        for key in data_stack.keys():
            data_arr[key] = data_stack[key][self.data_order]
        class_arr = class_stack[self.data_order]  # repeat for class matrix

        # remove missing values
        bad_datapoints = []
        for key, data in data_arr.items():
            for row_num, row in enumerate(data):
                for el in row:
                    if (
                        # only floats can be filtered using isnan()
                        (isinstance(el, np.floating) and np.isnan(el))
                        # filter null strings
                        or (isinstance(el, str) and not el)
                        # Nonetype values are only valid in pos_class and neg_class
                        or el is None
                    ) and (row_num not in bad_datapoints):
                        bad_datapoints.append(row_num)

        # propagate row removals to each dataset
        for row_num in bad_datapoints:
            for key in data_arr.keys():
                # datasets
                data_arr[key].pop(row_num)
            class_arr.pop(row_num)  # class matrix

        # store arranged, cleaned, normalized values
        self.data_arr = data_arr
        self.class_arr = class_arr

        # assign labels to targets
        # assign True if corresponding class entry is M1.0 or higher
        # self.labels = [1, 0, 1, 1, 0, ... 0, 1] etc
        # self.labels[65] = 1
        # self.daat[65] = {...} --> class from labels
        self.labels = [
            (
                1
                if isinstance(x[-1], str)
                and (
                    x[-1].startswith("X")
                    or (x[-1].startswith("M") and float(x[-1].removeprefix("M")) >= 1.0)
                )
                else 0
            )
            for x in self.class_arr
        ]

    # feature_creation() function takes as input the features set label (e.g. FS-I, FS-II, FS-III, FS-IV)
    # and creates 2 D array of corresponding features
    # for both positive and negative observations.
    # this array will be input to the svm model
    # For instance, if the input is FS-I, the output is a 2-d array with features corresponding to
    # FS-I for both negative and positive class observations
    def feature_creation(self, fs_value: list) -> Fset:
        fs_value = [i for i in fs_value if i]  # remove false values
        feature_set = None

        if "FS-I" in fs_value:
            # select subset
            fs_1 = self.data_arr["features_main_timechange"][:, :18]

            # append current feature subset to existing feature set
            feature_set = (
                fs_1 if feature_set is None else np.hstack((feature_set, fs_1))
            )

        if "FS-II" in fs_value:
            fs_2 = self.data_arr["features_main_timechange"][:, 19:]

            feature_set = (
                fs_2 if feature_set is None else np.hstack((feature_set, fs_2))
            )

        if "FS-III" in fs_value:
            fs_3 = self.data_arr["features_historical"]

            feature_set = (
                fs_3 if feature_set is None else np.hstack((feature_set, fs_3))
            )

        if "FS-IV" in fs_value:
            fs_4 = self.data_arr["features_maxmin"]

            feature_set = (
                fs_4 if feature_set is None else np.hstack((feature_set, fs_4))
            )

        # Fset object: ndarray with additional name attr
        return Fset(feature_set, name=f"{fs_value}")

    # training() function trains a SVM classification model on input features and corresponding target
    def training(self, feature_set: np.ndarray, target: list) -> svm.SVC:
        model = svm.SVC()  # init SVM model
        model.fit(feature_set, target)
        return model

    # tss() function computes the accuracy of predicted outputs (i.e target prediction on test set)
    # using the TSS measure given in the document
    def tss(self, predicted: list, actual: list) -> float:
        tp, tn, fp, fn = 0, 0, 0, 0
        # count tp, tn, fp, fn
        for pred, acc in zip(predicted, actual):
            if pred and acc:
                tp += 1
            elif not (pred or acc):
                tn += 1
            elif pred and not acc:
                fp += 1
            elif acc and not pred:
                fn += 1

        if not ((tp + fn) and (fp + tn)):  # if denominator == 0
            print(
                "ZeroDivisionError: TSS cannot be calculated due to no positives or "
                "no negatives being present in the set, returning accuracy percentage instead..."
            )
            return (tp + tn) / (tp + tn + fp + fn)

        return tp / (tp + fn) - fp / (fp + tn)

    # cross_validation() function splits the data into train and test splits,
    # Use k-fold with k=10
    # the svm is trained on training set and tested on test set
    # the output is the average accuracy across all train test splits.
    def cross_validation(
        self, feature_set: Fset, target_set: list, k: int = 10
    ) -> np.floating:
        # create k folds
        folds = KFolds(feature_set, target_set, k)

        scores = []
        c_matrices = []
        for i, _ in enumerate(folds.Y_folds):
            fold = folds.get_train_test(i)  # select fold
            # call training function
            model = self.training(fold.train_X, fold.train_Y)
            pred_y = model.predict(fold.test_X)
            # call tss function and add to list of scores
            scores.append(self.tss(predicted=pred_y, actual=fold.test_Y))
            # get confusion matrix and add to list
            c_matrices.append(confusion_matrix(fold.test_Y, pred_y))

        tss_avg = np.mean(scores)  # get mean tss
        tss_std = np.std(scores)  # get std dev tss
        # BEGIN CREATE PLOT
        print(f"Average TSS, std dev for {feature_set.name}: {tss_avg}, {tss_std}")
        fig, axs = plt.subplots(1, 2)
        fig.suptitle(f"TSS & confusion matrix for {self.dir}: {feature_set.name}")
        axs[0].plot(range(len(scores)), scores, "-o")
        axs[0].set_ylabel("TSS Score")
        axs[0].set_xlabel("Fold Number")
        axs[0].set_title("TSS score for each fold")
        axs[0].annotate(
            "Avg TSS= %.6f\n" % round(tss_avg, 6) + "Std dev= %.6f" % round(tss_std, 6),
            xy=(1, 0),
            xycoords="axes fraction",
            xytext=(-20, 20),
            textcoords="offset pixels",
            horizontalalignment="right",
            verticalalignment="bottom",
        )  # add annotation for avg and std dev

        c = np.mean(np.stack(c_matrices), axis=0)  # get average confusion matrix
        ConfusionMatrixDisplay(c).plot(ax=axs[1])  # plot confusion matrix
        # END CREATE PLOT

        return np.average(scores)  # return average tss score


# feature_experiment() function executes experiments with all 4 feature sets.
# svm is trained (and tested) on 2010 dataset with all 4 feature set combinations
# the output of this function is :
#  1) TSS average scores (mean std) for k-fold validation printed out on console.
#  2) Confusion matrix for TP, FP, TN, FN. See assignment document
#  3) A chart showing TSS scores for all folds of CV.
#     This means that for each fold, compute the TSS score on test set for that fold, and plot it.
#     The number of folds will decide the number of points on this chart (i.e 10)
#
# Above 3 charts are produced for all feature combinations
#  4) The function prints the best performing feature set combination
def feature_experiment() -> list:
    print("------------------")
    print("Feature experiment")
    print("------------------")

    s_2015 = SVM("data-2010-15")
    s_2024 = SVM("data-2020-24")

    s_2015.preprocess()
    s_2024.preprocess()

    scores = []
    fsets = ["FS-I", "FS-II", "FS-III", "FS-IV"]
    set_keys = [
        [
            bool(int(j)) for j in str(format(i, "004b"))
        ]  # generate list of booleans in binary order
        for i in range(1, 2 ** len(fsets))  # 2^n combinations
    ]  # get keys to access all combinations of fsets
    for set_key in set_keys:
        # i.e. feat = ['FS-I', False, False, 'FS-IV']
        feat = [a and b for a, b in zip(set_key, fsets)]
        X = s_2015.feature_creation(feat)  # create training feature set
        X_predict = s_2024.feature_creation(feat)  # create test feature set
        y = s_2015.labels  # training set actual results
        y_predict = s_2024.labels  # test set actual results

        model = s_2015.training(feature_set=X, target=y)  # train model
        model.predict(X_predict)  # predict 2024 (test) data
        scores.append((feat, model.score(X_predict, y_predict)))  # track score

        # cross validate 2015 (training) data
        s_2015.cross_validation(s_2015.feature_creation(feat), s_2015.labels)

    # get best feature set: find max tss score
    # choose smaller set if multiple sets have the same tss score
    _best_set = max(scores, key=lambda x: (x[1], -len(x[0])))[0]
    print(f"The best performing set is {[i for i in _best_set if i]}")
    return _best_set


# data_experiment() function executes 2 experiments with 2 data sets.
# svm is trained (and tested) on both 2010 data and 2020 data
# the output of this function is :
#  1) TSS average scores for k-fold validation printed out on console.
#  2) Confusion matrix for TP, FP, TN, FN. See assignment document
#  3) A chart showing TSS scores for all folds of CV.
#     This means that for each fold, compute the TSS score on test set for that fold, and plot it.
#     The number of folds will decide the number of points on this chart (i.e. 10)
# above 3 charts are produced for both datasets
# feature set for this experiment should be the
# best performing feature set combination from feature_experiment()
def data_experiment(_best_set: list) -> None:
    print("---------------")
    print("Data experiment")
    print("---------------")

    s_2015 = SVM("data-2010-15")
    s_2024 = SVM("data-2020-24")

    s_2015.preprocess()
    s_2024.preprocess()

    s_2015.cross_validation(s_2015.feature_creation(_best_set), s_2015.labels)
    s_2024.cross_validation(s_2024.feature_creation(_best_set), s_2024.labels)


# below should be your code to call the above classes and functions
# with various combinations of feature sets
# and both datasets
if __name__ == "__main__":
    best_set = feature_experiment()
    data_experiment(best_set)

    plt.show()
