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
    """Gini coefficient normalized."""
    return gini(a, p) / gini(a, a)


train = pd.read_csv("data/train.csv", index_col=0)
test = pd.read_csv("data/test.csv", index_col=0)

transformations = []
for c in train.columns:
    if c == "target":
        continue
    elif "cat" in c:
        transformations.append(
            ([c], sklearn.preprocessing.LabelBinarizer()))
    elif not ("cat" in c or "bin" in c):
        transformations.append(
            ([c], sklearn.preprocessing.StandardScaler()))
mapper = DataFrameMapper(transformations)

X0 = mapper.fit_transform(train.drop("target", axis=1))
X_test0 = mapper.fit_transform(test)
y0 = np.array(train["target"])

dim = 99999
X = X0[:dim, :]
X_test = X_test0[:dim, :]
y = y0[:dim]

clf = linear_model.LinearRegression()
# clf = neighbors.KNeighborsRegressor(n_neighbors=5, weights="distance")
clf.fit(X, y)

y_p = clf.predict(X)

y_test_p = clf.predict(X_test)
for i in range(len(y_test_p)):
    y_test_p[i] = max(0, min(1, y_test_p[i]))

prediction = pd.DataFrame(
    index=test.index[:dim],
    data=np.round(y_test_p, 3),
    columns=["target"])
prediction.to_csv("data/prediction.csv")
