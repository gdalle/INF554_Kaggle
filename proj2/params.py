# External imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import scipy.stats as st
import pickle as pk
from sklearn import linear_model, tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss, make_scorer
from sklearn.model_selection import KFold, GridSearchCV

# Internal imports
import importlib
import extraction as ex
import features as feat
ex = importlib.reload(ex)
feat = importlib.reload(feat)

# Read data

train = ex.read_train()
test = ex.read_test()
members = ex.read_members()
useful_msno = set.union(
    set(train.index.unique()),
    set(test.index.unique())
)

transactions_train_data = pd.read_csv(
    "/tmp/kaggle/junk/transactions_train_data.csv",
    index_col=0
)
transactions_data = pd.read_csv(
    "/tmp/kaggle/junk/transactions_data.csv",
    index_col=0
)

data_list_train = [transactions_train_data]
data_list = [transactions_data]

# Add the data to the train set and test dataframes
train_full = train.join(data_list_train, how="inner")
test_full = test.join(data_list, how="inner")

# Keep only the features we want
features = [c for c in test_full.columns if not "payment_method" in c]
train_filtered, test_filtered = feat.select_features(train_full, test_full, features)

# Normalize the columns
train_filtered, test_filtered = feat.normalize_features(train_filtered, test_filtered)

def log_loss_score_func(y_true, y_pred):
    return log_loss(
        y_true, y_pred, labels=[0, 1],
        eps=np.power(10., -15), normalize=True)

log_loss_scorer = make_scorer(
    score_func=log_loss_score_func,
    greater_is_better=False,
    needs_proba=True
)

x = np.array(train_filtered.drop("is_churn", axis=1))
y = np.array(train_filtered["is_churn"])
xt = np.array(test_filtered)

xgbclf = xgb.XGBClassifier()

k = 5
param_grid = {
    "n_estimators": np.round(0.5 * np.power(10, np.linspace(2, 3, k))).astype(int),
    "max_depth": np.arange(2, 6),
    "learning_rate": np.power(10, np.linspace(-1.3, -0.3, k))
}

total_tasks=1
for k in param_grid:
    total_tasks *= len(param_grid[k])
print("Total tasks for grid search :", total_tasks, "\n")

gs = GridSearchCV(
    estimator=xgbclf,
    param_grid=param_grid,
    cv=3,
    scoring=log_loss_scorer,
    n_jobs=4, verbose=3)  

gs.fit(x, y, eval_metric="logloss") 

pk.dump(gs.cv_results_, open("GS_results", "wb"))
pk.dump(gs.best_estimator_, open("GS_estimator", "wb"))
best_xgbclf = gs.best_estimator_

# Use best classifier for prediction
best_xgbclf.fit(x, y, eval_metric="logloss")
yt = best_xgbclf.predict_proba(xt)[:, 1]

# Zero prediction as baseline
percentage_churn = train_filtered["is_churn"].sum() / len(train_filtered)
test["is_churn"] = np.random.rand(len(test)) * percentage_churn
# For users on which we have more info, use it
test.loc[test_filtered.index, ["is_churn"]] = yt.reshape(-1, 1)

# Save as csv
submission = test.loc[:, ["is_churn"]]
submission.to_csv("data/submission_gs.csv")