import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as f
from torch.utils.data import RandomSampler, DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    Grayscale,
    Resize,
    RandomCrop,
    ToTensor,
)

SENSITIVE_COL_IDX = 18

class LogisticRegression(nn.Module):
    def __init__(self, input_dimension):
        super().__init__()
        self.linear = nn.Linear(input_dimension, 1)

    def forward(self, x):
        # y_pred = torch.sigmoid(self.linear(x))
        y_pred = self.linear(x)
        return y_pred


def crime_desc_encoding(df, column_name):
    categories = {
        "Abuse": {"incl": ["Abuse", "neglect", "cruelty"], "excl": []},
        "Sex": {"incl": ["sex", "lewd", "nude", "prostitution", "voyeur"], "excl": []},
        "Assault": {"incl": ["battery", "batt", "Assault"], "excl": []},
        "Aggravated": {"incl": ["agg", "aggr", "aggravated"], "excl": []},
        "Child": {"incl": ["child", "minor"], "excl": []},
        "Fleeing": {"incl": ["eluding", "fleeing"], "excl": []},
        "Drug": {
            "incl": [
                "traff",
                "dui",
                "d.u.i.",
                "intoxication",
                "under the influence",
                "intoxicated",
                "del",
                "deliver",
                "cannabis",
                "alcohol",
                "heroin",
                "morphine",
                "alprazolam",
                "cocaine",
                "oxycodone",
                "contr sub",
                "drug",
                "pyrrolidinobutiophenone",
                "rx without rx",
                "controlled substance",
                "amobarbital",
                "clonazepam",
                "methylene",
                "tobacco",
                "anabolic steroid",
                "buprenorphine",
                "carisoprodol",
                "diazepam",
                "fentanyl",
                "lorazepam",
                "mdma",
                "phentermine",
                "benzylpiperazine",
                "butylone",
                "ethylone",
                "hydrocodone",
                "meth",
                "amphetamine",
                "ecstasy",
            ],
            "excl": [],
        },
        "Armed": {
            "incl": ["armed", "gun", "shoot", "weapon", "firearm", "throw"],
            "excl": ["no weapon", "w/o weapon", "w/o deadly weapon"],
        },
        "Homicide": {"incl": ["murder", "manslaughter", "homicide"], "excl": []},
        "Theft": {"incl": ["theft", "robbery", "burgl", "burglary"], "excl": []},
        "Trespass": {"incl": ["trespass", "tresspass"], "excl": []},
        "Fraud": {"incl": ["forge", "fraud", "altered"], "excl": []},
        "Traffic violation": {
            "incl": [
                "reckless",
                "suspended license",
                "vehicle",
                "driving",
                "hit and run",
            ],
            "excl": [],
        },
    }

    def assign_cat(x, incl, excl):
        if not pd.isna(x):
            return int(
                any(kw.lower() in x.lower() for kw in incl)
                and not any(kw.lower() in x.lower() for kw in excl)
            )
        return 0

    c_desc = pd.DataFrame()
    for category, keywords in categories.items():
        incl, excl = keywords.values()
        c_desc[f"{column_name}_{category}"] = df[column_name].apply(
            lambda x: assign_cat(x, incl, excl)
        )
    df = pd.concat([df, c_desc], axis=1)
    df = df.drop(column_name, axis=1)
    return df


def preprocess(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    # columns containing useless/redundant info to be dropped
    drop_cols = [
        "id",
        "name",
        "first",
        "last",
        "dob",
        "c_case_number",
        "num_r_cases",
        "r_case_number",
        "num_vr_cases",
        "vr_case_number",
        "v_type_of_assessment",
        "type_of_assessment",
        "compas_screening_date",
        "c_arrest_date",
        "screening_date",
        "v_screening_date",
    ]
    df = df.drop(columns=drop_cols)

    # remove samples where jail in/out date is missing
    df = df[df["c_jail_out"].notna()]
    df = df[df["c_jail_in"].notna()]

    # Find time spent in jail
    c_jail_time = (
        pd.to_datetime(df["c_jail_out"]) - pd.to_datetime(df["c_jail_in"])
    ).dt.days

    # time spent in jail for recurring offense
    r_jail_time = (
        pd.to_datetime(df["r_jail_out"]) - pd.to_datetime(df["r_jail_in"])
    ).dt.days.fillna(0)
    df["total_jail_time"] = c_jail_time + r_jail_time  # total time in jail

    # time between case in question and reoffense (if applicable)
    df["r_time_to_reoffend"] = (
        pd.to_datetime(df["r_offense_date"]) - pd.to_datetime(df["c_offense_date"])
    ).dt.days

    # time between case in quesrion and violent reoffense (if applicable)
    df["vr_time_to_reoffend"] = (
        pd.to_datetime(df["vr_offense_date"]) - pd.to_datetime(df["c_offense_date"])
    ).dt.days

    # take inverse proportion and set samples with no reoffense as zero
    # thereby making no reoffense the lowest value, and a reoffense shortly after
    # the original defense date the highest value
    df["r_time_to_reoffend"] = 1 / (df["r_time_to_reoffend"] + 1)
    df["vr_time_to_reoffend"] = 1 / (df["vr_time_to_reoffend"] + 1)
    df["r_time_to_reoffend"] = df["r_time_to_reoffend"].fillna(0)
    df["vr_time_to_reoffend"] = df["vr_time_to_reoffend"].fillna(0)

    # drop columns used to calculate the above features
    df = df.drop(
        [
            "c_jail_in",
            "c_jail_out",
            "c_offense_date",
            "r_offense_date",
            "vr_offense_date",
            "r_jail_in",
            "r_jail_out",
        ],
        axis=1,
    )

    # one-hot encode the race, sec, and age categories
    race_one_hot = pd.get_dummies(df["race"], prefix="race", dtype=int)
    sex_one_hot = pd.get_dummies(df["sex"], prefix="sex", dtype=int)
    age_cat_one_hot = pd.get_dummies(df["age_cat"], prefix="age_cat", dtype=int)

    # add one-hots to feature matrix and remove original data
    df = pd.concat([df, race_one_hot, sex_one_hot, age_cat_one_hot], axis=1)
    df = df.drop(["race", "sex", "age_cat"], axis=1)

    for prefix in ["c", "r"]:
        # analyze text in crime description to categorize crime
        # i.e. "domestic violence/battery" is categorized into "domestic" and "assault"
        df = crime_desc_encoding(df, f"{prefix}_charge_desc")

        # encode charge degree
        df[f"{prefix}_charge_degree"] = df[f"{prefix}_charge_degree"].map(
            {"O": 0, "M": 1, "F": 2}
        )

    # encode violent reoffense charge degree
    df["vr_charge_degree"] = (
        df["vr_charge_degree"]
        .map(
            {
                "(MO3)": 1,
                "(M2)": 2,
                "(M1)": 3,
                "(F7)": 4,
                "(F6)": 5,
                "(F5)": 6,
                "(F4)": 7,
                "(F3)": 8,
                "(F2)": 9,
                "(F1)": 10,
            }
        )
        .fillna(0)
    )
    # df = df.drop(['vr_charge_degree'], axis=1)

    # encode violent reoffense charge description
    df = crime_desc_encoding(df, "vr_charge_desc")

    # encode score text as labels
    df["v_score_text"] = df["v_score_text"].map({"Low": 0, "Medium": 0, "High": 1})
    df["score_text"] = df["score_text"].map({"Low": 0, "Medium": 0, "High": 1})

    # fill missing values with median (don't want to remove, ~10% of dataset)
    col = df["days_b_screening_arrest"]
    df["days_b_screening_arrest"] = col.fillna(col.median())

    # fill missing values with median (don't want to remove, ~10% of dataset)
    col = df["c_days_from_compas"]
    df["c_days_from_compas"] = col.fillna(col.median())

    # drop as 80% of the dataset is missing values in this column
    df = df.drop("r_days_from_arrest", axis=1)

    # redundant: age bins already included
    df = df.drop("age", axis=1)

    # remove entries with missing data in this category
    df = df[df["is_recid"] != -1]

    # remove all remaining rows with NaN values
    df = df.dropna()

    # extract labels from data
    labels = df["score_text"].to_numpy()
    df = df.drop(["score_text"], axis=1)

    # save index of sensitive feature (race = african-american)
    global SENSITIVE_COL_IDX
    SENSITIVE_COL_IDX = df.columns.get_loc("race_African-American")

    # normalize data
    scaler = MinMaxScaler()

    # separate data into labels and features
    features = scaler.fit_transform(df.to_numpy())

    # convert results to Tensors
    return torch.from_numpy(features).to(dtype=torch.float), torch.from_numpy(
        labels
    ).to(dtype=torch.float)


def equalize_dist(
    X: torch.Tensor, y: torch.Tensor, feature_idx: int = SENSITIVE_COL_IDX
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Equalizes the distribution of a dataset with respect to a single (sensitive) feature by
    iteratively removing rows until the dataset is balanced w.r.t. the feature.
    X: Feature tensor
    y: Label tensor
    feature_idx: index of target feature
    return: X and y with random rows removed to achieve an equal distribution.
    """
    # init subtensor with just sensitive feature and labels
    pairs = torch.cat((X[:, feature_idx].unsqueeze(1), y), dim=1)
    pairs = pairs[pairs[:, 1] == 1]  # get all rows where sensitive feature is 1
    mask = torch.ones(X.shape[0], dtype=bool)  # init mask (1d tensor of ones)

    while pairs.shape[0]:  # while tensor is not empty
        positive_labels = pairs[:, 1].sum()  # current amount of positive labels

        # amount of positive labels needed for an equal distribution
        equal_dist_amt = pairs.shape[0] / 2

        idx = np.random.randint(0, pairs.shape[0])  # init random row number

        if positive_labels == equal_dist_amt:
            # Break if distribution of 1's and 0's in the sub-tensor is equal
            break

        # when there are more 0's than 1's in the dataset
        elif positive_labels < equal_dist_amt:
            # select a random row until we find a row where the class is 0
            while pairs[idx, 1] != 0:
                idx = np.random.randint(0, pairs.shape[0])

        # when there are more 1's than 0's in the dataset
        elif positive_labels > equal_dist_amt:
            # select a random row until we find a row where the class is 1
            while pairs[idx, 1] != 1:
                idx = np.random.randint(0, pairs.shape[0])

        # remove specified row from tensor
        pairs = torch.cat((pairs[:idx], pairs[idx + 1 :]), dim=0)
        mask[idx] = False  # mark row number for removal

    return X[mask], y[mask]


def equalized_odds(
    X: torch.Tensor,
    y: torch.Tensor,
    y_pred: torch.Tensor,
    feature_idx: int = SENSITIVE_COL_IDX,
) -> float:
    """
    Calculates the equalized odds with respect to a feature specified by `feature_idx`
    :param X: Feature tensor
    :param y: actual labels
    :param y_pred: predicted labels
    :param feature_idx: index of sensitive feature
    :return: equalized odds
    """
    y = y.squeeze(1)
    mask = X[:, feature_idx].bool()
    y_pred_unpriv = y_pred[mask]
    y_unpriv = y[mask]
    tn, fp, fn, tp = confusion_matrix(y_unpriv, y_pred_unpriv).ravel()
    unpriv_tpr = tp / (tp + fn)
    unpriv_fnr = fn / (tp + fn)

    y_pred_priv = y_pred[~mask]
    y_priv = y[~mask]
    tn, fp, fn, tp = confusion_matrix(y_priv, y_pred_priv).ravel()
    priv_tpr = tp / (tp + fn)
    priv_fnr = fn / (tp + fn)

    return abs(priv_tpr - unpriv_tpr) + abs(priv_fnr - unpriv_fnr)


if __name__ == "__main__":
    data = pd.read_csv("data/compas-scores.csv")

    X, y = preprocess(data)
    y = y.unsqueeze(1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # pd.DataFrame(X).to_csv('data/features.csv')
    # pd.DataFrame(y).to_csv('data/labels.csv')

    # Train
    l_rate = 0.01
    epochs = 10000
    net = LogisticRegression(X.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=l_rate)

    print("Training on complete training set")
    for epoch in range(epochs):
        outputs = net(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Test
    print("Testing...")
    total = 0
    correct = 0
    y_pred = []
    # with torch.no_grad():
    #     for sample, real_label in zip(X_test, y_test):
    #         output = net(sample)
    #         predicted = float(output.data.item() > 0.5)
    #         y_pred.append(predicted)
    #         correct += predicted == real_label.item()
    #         total += 1

    with torch.no_grad():
        outputs = net(X_test)
        y_pred = (outputs > 0.5).float()
        correct = (y_pred == y_test).sum().item()
        total = y_test.size(0)

    y_pred = torch.Tensor(y_pred)
    e_odds = equalized_odds(X_test, y_test, y_pred)

    print(f"accuracy: {correct / total}")
    print(f"equalized odds: {e_odds}\n\n")

    print("Overall High Likelihood analysis")
    print("race -- tp + fp : tn + fn, pos/neg")
    for r, i in [['african-american', 0], ['asian', 1], ['cauc', 2], ['hisp', 3], ['native', 4], ['other', 5]]:
        mask = X_test[:, SENSITIVE_COL_IDX+i].bool()
        y_mask = y_test.squeeze(1)[mask]
        y_pred_mask = y_pred.squeeze(1)[mask]
        cm = confusion_matrix(y_mask, y_pred_mask).ravel()
        if len(cm) == 4:
            tn, fp, fn, tp = cm
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
            for i, j in zip(list(y_mask), list(y_pred_mask)):
                tn += int(not i and not j)
                fp += int(j and not i)
                fn += int(i and not j)
                tp += int(i and j)

        print(f"{r} -- {tp} + {fp} : {tn} + {fn}, {(tp+fp)/(tn+fn)}")

    # print("Unreasonably high likelihood analysis")
    # is_recid = SENSITIVE_COL_IDX - 10
    # for r, i in [['african-american', 0], ['asian', 1], ['cauc', 2], ['hisp', 3], ['native', 4], ['other', 5]]:
    #     mask1 = X_test[:, SENSITIVE_COL_IDX+i].bool()
    #     mask2 = X_test[:, is_recid].bool()
    #     mask = mask1 & mask2
    #
    #     y_mask = y_test.squeeze(1)[mask]
    #     y_pred_mask = y_pred.squeeze(1)[mask]
    #     cm = confusion_matrix(y_mask, y_pred_mask).ravel()
    #     if len(cm) == 4:
    #         tn, fp, fn, tp = cm
    #     else:
    #         tn, fp, fn, tp = 0, 0, 0, 0
    #         for i, j in zip(list(y_mask), list(y_pred_mask)):
    #             tn += int(not i and not j)
    #             fp += int(j and not i)
    #             fn += int(i and not j)
    #             tp += int(i and j)
    #
    #     print(f"{r} -- {(tp+fp)} : {(tn+fn)}, {(tp+fp)/(tn+fn)}")

    #################################################################
    # Train with equalized distribution of sensitive feature (race) #
    #################################################################

    X_train_eq, y_train_eq = equalize_dist(X_train, y_train)
    net = LogisticRegression(X.shape[1])
    print("Training on equal distribution training set...")
    for epoch in range(epochs):
        _y_pred = net(X_train_eq)
        loss = criterion(_y_pred, y_train_eq)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Test
    print("Testing...")
    y_pred = []
    total = 0
    correct = 0
    with torch.no_grad():
        outputs = net(X_test)
        y_pred = (outputs > 0.5).float()
        correct = (y_pred == y_test).sum().item()
        total = y_test.size(0)

    y_pred = torch.Tensor(y_pred)
    e_odds = equalized_odds(X_test, y_test, y_pred)

    print(f"accuracy: {correct / total}")
    print(f"equalized odds: {e_odds}")

    # print("Overall High Likelihood analysis")
    # for r, i in [['african-american', 0], ['asian', 1], ['cauc', 2], ['hisp', 3], ['native', 4], ['other', 5]]:
    #     mask = X_test[:, SENSITIVE_COL_IDX+i].bool()
    #     y_mask = y_test.squeeze(1)[mask]
    #     y_pred_mask = y_pred[mask]
    #     cm = confusion_matrix(y_mask, y_pred_mask).ravel()
    #     if len(cm) == 4:
    #         tn, fp, fn, tp = cm
    #     else:
    #         tn, fp, fn, tp = 0, 0, 0, 0
    #         for i, j in zip(list(y_mask), list(y_pred_mask)):
    #             tn += int(not i and not j)
    #             fp += int(j and not i)
    #             fn += int(i and not j)
    #             tp += int(i and j)
    #
    #     print(f"{r} -- {(tp+fp)} : {(tn+fn)}, {(tp+fp)/(tn+fn)}")

    # print("Unreasonably high likelihood analysis")
    # is_recid = SENSITIVE_COL_IDX - 10
    # for r, i in [['african-american', 0], ['asian', 1], ['cauc', 2], ['hisp', 3], ['native', 4], ['other', 5]]:
    #     mask1 = X_test[:, SENSITIVE_COL_IDX+i].bool()
    #     mask2 = X_test[:, is_recid].bool()
    #     mask = mask1 & mask2
    #
    #     y_mask = y_test.squeeze(1)[mask]
    #     y_pred_mask = y_pred[mask]
    #     cm = confusion_matrix(y_mask, y_pred_mask).ravel()
    #     if len(cm) == 4:
    #         tn, fp, fn, tp = cm
    #     else:
    #         tn, fp, fn, tp = 0, 0, 0, 0
    #         for i, j in zip(list(y_mask), list(y_pred_mask)):
    #             tn += int(not i and not j)
    #             fp += int(j and not i)
    #             fn += int(i and not j)
    #             tp += int(i and j)
    #
    #     print(f"{r} -- {(tp+fp)} : {(tn+fn)}, {(tp+fp)/(tn+fn)}")
