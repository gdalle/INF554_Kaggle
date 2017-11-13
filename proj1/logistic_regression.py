import pandas as pd
import numpy as np
from sklearn import linear_model
from evaluation import *
from data_processing import *
from sklearn import preprocessing

# Read csv files

print("=================")
print("Loading data ...")

train = pd.read_csv("data/train.csv", index_col=0)
test = pd.read_csv("data/test.csv", index_col=0)

print("Done\n")

# Feature engineering

print("=================")
print("Transforming categorical variables ...")
# Transform categorical variables into dummy binary variables
train = binarize_cat(train)
test = binarize_cat(test)
print("Done\n")


print("=================")
print("Processing data ...")
# Turn DataFrames into arrays for scikit-learn
# Forget the target column in X
X0 = np.array(train.drop("target", axis=1))
y0 = np.array(train["target"])


# Splitting training set into training and validation set
size_training = int(.8*X0.shape[0])
size_validation = X0.shape[0]-size_training
train_validate = np.random.choice(np.arange(X0.shape[0]),size=size_training+size_validation,replace=False)
X_train = X0[train_validate[:size_training],]
y_train = y0[train_validate[:size_training]]
X_validate = X0[train_validate[size_training:size_training+size_validation],]
y_validate = y0[train_validate[size_training:size_training+size_validation]]
X_test = np.array(test)

# Imputing missing values

imputer = preprocessing.Imputer(missing_values=-1).fit(X_train)

X_train = imputer.transform(X_train)
X_validate = imputer.transform(X_validate)
X_test = imputer.transform(X_test)


# Scaling data

scaler = preprocessing.StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_validate = scaler.transform(X_validate)
X_test = scaler.transform(X_test)

print("Done\n")


print("=================")
print("Learning ...")
# Choose and train machine learning algorithm
clf = linear_model.LogisticRegression(C=.5)
# Fit on the training set
clf.fit(X_train, y_train)

print("=================")
print("Predicting for evaluation ...")
# Predict results on validation set
y_p = clf.predict_proba(X_validate)[:,1]
print("Done\n")

print("=================")
print("Evaluating ...")
print("Normalized Gini index : {}".format(gini_normalized(y_validate,y_p)))
print("Done\n")


print("=================")
print("Predicting for submission ...")
y_test_p = clf.predict_proba(X_test)[:,1]
prediction = pd.DataFrame(
    index=test.index,
    data=np.round(y_test_p, 3),
    columns=["target"])
prediction.to_csv("data/submission.csv")
print("Done\n")