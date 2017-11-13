"""Perform machine learning."""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import dummy
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss, make_scorer
import extraction as ex
import features as feat

# MAIN

# User dataframes
train = ex.read_train()
test = ex.read_test()
members = ex.read_members()

# Transactions and user logs dataframes
transactions = ex.read_transactions(
    train, test, max_lines=10**6, split_dates=False)
user_logs = ex.read_user_logs(
    train, test, max_lines=10**6, split_dates=False)

# Users on which we have info
train_useful = feat.get_useful_users(train, members, transactions, user_logs)
test_useful = feat.get_useful_users(test, members, transactions, user_logs)

# Add features taken from members table
train_useful = feat.add_members_info(train_useful, members)
test_useful = feat.add_members_info(test_useful, members)

# Here comes the machine learning

print("\nPREDICTION\n")

# Conversion into arrays for scikit-learn
x = np.array(train_useful.drop("is_churn", axis=1))
y = np.array(train_useful["is_churn"])
xt = np.array(test_useful)

# Linear Regression fitting
clf = linear_model.LogisticRegression()
clf.fit(x, y)

# Cross-validation
log_loss_scorer = make_scorer(
    score_func=log_loss,
    eps=np.power(10., -15),
    normalize=True,
    greater_is_better=False,
    needs_proba=True
)
scores = cross_val_score(
    estimator=clf,
    X=x,
    y=y,
    cv=5,
    scoring=log_loss_scorer
)
print("CV score (log-loss) : {}".format(scores.mean()))

# Prediction
# Compute the probability of belonging to class 1 (and not 0)
yt = clf.predict_proba(xt)[:, 1]

# Zero prediction as baseline
test["is_churn"] = 0.
# For users on which we have more info, use it
test.loc[test_useful.index, ["is_churn"]] = yt.reshape(-1, 1)

# Save as csv
submission = test.loc[:, ["is_churn"]]
submission.to_csv("data/submission.csv")
