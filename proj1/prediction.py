"""Reading data and creating output."""

import pandas as pd
import numpy as np
import sklearn.preprocessing
from sklearn import linear_model, neighbors, svm, tree
from sklearn.model_selection import cross_val_score
from sklearn_pandas import DataFrameMapper


def gini(actual, pred, cmpcol=0, sortcol=1):
    """Compute Gini coefficient."""
    assert(len(actual) == len(pred))
    all = np.asarray(
        np.c_[actual, pred, np.arange(len(actual))],
        dtype=np.float)
    all = all[np.lexsort((all[:, 2], - 1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(a, p):
    """Compute normalized Gini coefficient."""
    return gini(a, p) / gini(a, a)


def binarize_cat(data):
    """Return a new DataFrame with categorical variables encoded as 0/1."""
    for c in data.columns:
        if "cat" in c:
            # Get 0/1 columns associated with categories
            dummies = pd.get_dummies(data[c])
            # Change column names to form var==1, var==2...
            dummies.columns = [c + "=" + str(dc) for dc in dummies.columns]
            # Concatenate to original data
            data = data.drop(c, axis=1).join(dummies)
    return data


def filter_shitty_columns(data, max_missing=1, display=False):
    """Find out which columns have too much missing data."""
    total = len(data)
    shitty_columns = []
    for c in data.columns:
        count = data[c].value_counts()
        # Compute number of missing entries
        if -1 in count.index:
            missing = count.loc[-1]
            missing_percent = np.round(missing / total * 100, 3)
        else:
            missing = 0
            missing_percent = 0
        if display:
            # Print info
            print()
            print("=================")
            print(c)
            print("{} different values".format(len(data[c].unique())))
            print("{} missing entries, i.e. {} %".format(
                missing, missing_percent))
            print(count.head())
            print("=================")
        # If the percentage of missing data is too high, keep track
        if missing_percent > max_missing:
            shitty_columns.append(c)

    for sc in shitty_columns:
        data = data.drop(sc, axis=1)
    return data


# Read csv files
train = pd.read_csv("data/train.csv", index_col=0)
test = pd.read_csv("data/test.csv", index_col=0)

# Throw away columns with missing data
train = filter_shitty_columns(train)
test = filter_shitty_columns(test)

# Feature engineering
# Transform categorical variables into dummy binary variables
train = binarize_cat(train)
test = binarize_cat(test)

# Dimensionality reduction ?

# Turn DataFrames into matrices for scikit-learn
# Forget the target column in X
X0 = np.array(train.drop("target", axis=1))
X_test0 = np.array(test)
y0 = np.array(train["target"])

# Reduce number of individuals (for testing purposes)
n = 100000
X = X0[:n, :]
X_test = X_test0[:n, :]
y = y0[:n]

# Choose and train machine learning algorithm
clf = linear_model.LogisticRegression()
# Fit on the training set
clf.fit(X, y)

# Predict results on train set
y_p = clf.predict(X)

# Predict results on test set
y_test_p = clf.predict(X_test)
for i in range(len(y_test_p)):
    # Force in interval [0, 1]
    y_test_p[i] = max(0, min(1, y_test_p[i]))

# Store the prediction in a DataFrame and export to csv
prediction = pd.DataFrame(
    index=test.index[:n],
    data=np.round(y_test_p, 3),
    columns=["target"])
prediction.to_csv("data/submission.csv")
