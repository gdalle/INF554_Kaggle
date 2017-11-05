"""Reading data and creating output."""

import pandas as pd
import numpy as np
import sklearn.preprocessing
from sklearn import linear_model, neighbors, svm, tree
from sklearn.model_selection import cross_val_score
from sklearn_pandas import DataFrameMapper


def gini(actual, pred, cmpcol=0, sortcol=1):
    """Gini coefficient."""
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
    """Normalized Gini coefficient."""
    return gini(a, p) / gini(a, a)


def binarize_cat(data):
    """Return a new DataFrame with categorical variables encoded as 0/1."""
    for c in data.columns:
        if "cat" in c:
            dummies = pd.get_dummies(data[c])
            dummies.columns = [c + "=" + str(dc) for dc in dummies.columns]
            data = data.drop(c, axis=1).join(dummies)
    return data


train = pd.read_csv("data/train.csv", index_col=0)
test = pd.read_csv("data/test.csv", index_col=0)


train = binarize_cat(train)
test = binarize_cat(test)

# Turn DataFrames into matrices for scikit-learn
X0 = np.array(train.drop("target", axis=1))
X_test0 = np.array(test)
y0 = np.array(train["target"])

# Reduce number of individuals (for testing purposes)
n = 99999999999999
X = X0[:n, :]
X_test = X_test0[:n, :]
y = y0[:n]

# Choose and train first machine learning algorithm
clf = linear_model.LinearRegression()
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
prediction.to_csv("data/prediction.csv")
