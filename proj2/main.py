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
transactions = ex.read_transactions(train, test, max_lines=10**8)
# user_logs = ex.read_user_logs(train, test, max_lines=10**7)

# Get useful users, appearing in the tables we are interested in
train_useful = feat.get_useful_users(
    train,
    members=members, transactions=transactions, user_logs=None)
test_useful = feat.get_useful_users(
    test,
    members=members, transactions=transactions, user_logs=None)

# Add members info
train_useful = feat.add_members_info(train_useful, members)
test_useful = feat.add_members_info(test_useful, members)

# Add transactions info
train_useful = feat.add_transactions_info(train_useful, transactions)
test_useful = feat.add_transactions_info(test_useful, transactions)

# Add user_logs info
# train_useful = feat.add_user_logs_info(train_useful, user_logs)
# test_useful = feat.add_user_logs_info(test_useful, user_logs)

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
    greater_is_better=True,
    needs_proba=True
)
scores = cross_val_score(
    estimator=clf,
    X=x,
    y=y,
    cv=3,
    scoring=log_loss_scorer
)
print("CV score (log-loss) : {}".format(scores.mean()))

# Prediction
# Compute the probability of belonging to class 1 (and not 0)
yt = clf.predict_proba(xt)[:, 1]
# yt = clf.predict(xt)

train_useful.head()

# Zero prediction as baseline
percentage_churn = train_useful["is_churn"].sum() / len(train_useful)
test["is_churn"] = np.random.rand(len(test)) * percentage_churn
# For users on which we have more info, use it
test.loc[test_useful.index, ["is_churn"]] = yt.reshape(-1, 1)

# Save as csv
submission = test.loc[:, ["is_churn"]]
submission.to_csv("data/submission.csv")
