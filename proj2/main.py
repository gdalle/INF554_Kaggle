"""Perform machine learning."""

# import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss, make_scorer
import importlib
import extraction as ex
import features as feat
ex = importlib.reload(ex)
feat = importlib.reload(feat)

# MAIN

# Read tables
train = ex.read_train()
test = ex.read_test()
members = ex.read_members()
useful_msno = set.union(
    set(train.index.unique()),
    set(test.index.unique())
)
transactions = ex.read_transactions(useful_msno=useful_msno, max_lines=10**6)
user_logs = ex.read_user_logs(useful_msno=useful_msno, max_lines=10**6)

# Get useful users
train_useful = feat.get_useful_users(train, members=members, transactions=transactions, user_logs=user_logs)
test_useful = feat.get_useful_users(test, members=members, transactions=transactions, user_logs=user_logs)

# Exploit the tables
members_data = feat.exploit_members(members)
transactions_data = feat.exploit_transactions(transactions)
user_logs_data = feat.exploit_user_logs(user_logs)

data_list = [members_data, transactions_data, user_logs_data]

# Add the data to the train set and test set
train_full = feat.add_data_to_users(train_useful, data_list)
test_full = feat.add_data_to_users(test_useful, data_list)

# Keep only the features we want
features = test_full.columns  # all of them
train_filtered, test_filtered = feat.select_features(train_full, test_full, features)

# Normalize the columns
train_filtered, test_filtered = feat.normalize_features(train_filtered, test_filtered)

# Here comes the machine learning

print("\nPREDICTION\n")

# Conversion into arrays for scikit-learn
x = np.array(train_filtered.drop("is_churn", axis=1))
y = np.array(train_filtered["is_churn"])
xt = np.array(test_filtered)

# Train a logistic regression
clf = linear_model.LogisticRegression()
# clf = linear_model.Ridge(alpha=0.)
clf.fit(x, y)

try:
    # Compute the probability of belonging to class 1 (and not 0)
    proba = True
    yt = clf.predict_proba(xt)[:, 1]
except AttributeError:
    # If impossible for this classifier, predict the value of the class
    # and restrict to the interval [0, 1]
    proba = False
    yt = clf.predict(xt)
    yt[yt < 0] = 0.
    yt[yt > 1] = 1.

# Perform cross-validation
log_loss_scorer = make_scorer(
    score_func=lambda y_true, y_pred: log_loss(
        y_true, y_pred, labels=[0, 1],
        eps=np.power(10., -15), normalize=True),
    greater_is_better=True,
    needs_proba=proba
)
scores = cross_val_score(
    estimator=clf,
    X=x,
    y=y,
    cv=5,
    scoring=log_loss_scorer
)
print("CV score (log-loss) : {}".format(scores.mean()))

# Zero prediction as baseline
percentage_churn = train_filtered["is_churn"].sum() / len(train_filtered)
test["is_churn"] = np.random.rand(len(test)) * percentage_churn
# For users on which we have more info, use it
test.loc[test_filtered.index, ["is_churn"]] = yt.reshape(-1, 1)

# Save as csv
submission = test.loc[:, ["is_churn"]]
submission.to_csv("data/submission.csv")
