import pandas as pd
import numpy as np
from sklearn import linear_model,tree
from evaluation import *
from data_processing import *
from sklearn import preprocessing


def logit_tree_fit(X, y, bin_cat_vars, num_vars):

    clf = tree.DecisionTreeClassifier(max_leaf_nodes=20)
    clf.fit(X[, :bin_cat_vars], y)

    leaves = clf.apply(X[, :bin_cat_vars])
    leaves_labels = set(leaves)
    leaves_map = np.zeros((X.shape[0]), len(leaves_labels))
    for i, l in enumerate(leaves):
        leaves_map[i, l] = 1

    imputers = []
    scalers = []
    logits = []

    for l in leaves_labels:
        myX = X[leaves_map[:, l], num_vars]
        my_y = y[leaves_map[:, l]]

        imp = preprocessing.Imputer(missing_values=-1).fit(myX)
        myX = imp.transform(myX)
        imputers.append(imp)

        scal = preprocessing.StandardScaler().fit(myX)
        myX = scal.transform(myX)
        scalers.append(scal)

        reg = linear_model.LogisticRegression()
        reg.fit(myX, my_y)
        logits.append(reg)

    return clf, leaves_labels, imputers, scalers, logits


def logit_tree_predict(X, bin_cat_vars, num_vars, clf, leaves_labels, imputers, scalers, logits):

    y = np.zeros(X.shape[0])

    leaves = clf.apply(X[:, bin_cat_vars])

    leaves_map = np.zeros((X_train.shape[0]), len(leaves_labels))
    for i, l in enumerate(leaves):
        leaves_map[i, l] = 1

    for l in leaves_labels:
        myX = X[leaves_map[:, l], num_vars]
        if myX.shape[0] > 0:
            myX = imputers[l].transform(myX)
            myX = scalers[l].transform(myX)
            y[leaves_map[:, l]] = logits[l].predict_proba(myX)[:,1]

    return y

# Read csv files

print("=================")
print("Loading data ...")

train = pd.read_csv("data/train.csv", index_col=0)
test = pd.read_csv("data/test.csv", index_col=0)

print("Done\n")

print("=================")
print("Processing data ...")
# Turn DataFrames into arrays for scikit-learn
# Forget the target column in X
X0 = train.drop("target", axis=1)

bin_cat_vars = is_cat_bin(X0)
num_vars = [i for i in range(X0.shape[1]) if i not in bin_cat_vars]

X0 = np.array(X0)
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

# Imputing missing values for categorical / binary features

bin_cat_imputer = preprocessing.Imputer(missing_values=-1,strategy="most_frequent").fit(X_train[:,bin_cat_vars])

X_train[:,bin_cat_vars] = bin_cat_imputer.transform(X_train[:,bin_cat_vars])

X_validate[:,bin_cat_vars] = bin_cat_imputer.transform(X_validate[:,bin_cat_vars])

X_test[:,bin_cat_vars] = bin_cat_imputer.transform(X_test[:,bin_cat_vars])

print("Done\n")


print("=================")
print("Learning ...")
clf, leaves_labels, imputers, scalers, logits = logit_tree_fit(X_train,y_train,bin_cat_vars,num_vars)

print("Done\n")

print("=================")
print("Predicting for evaluation ...")
# Predict results on validation set
y_p = logit_tree_predict(X_validate,bin_cat_vars,num_vars,clf,leaves_labels,imputers,scalers,logits)
print("Done\n")

print("=================")
print("Evaluating ...")
print("Normalized Gini index : {}".format(gini_normalized(y_validate,y_p)))
print("Done\n")


print("=================")
print("Predicting for submission ...")
y_test_p = clf.predict_proba(X_test)
prediction = pd.DataFrame(
    index=test.index,
    data=np.round(y_test_p, 3),
    columns=["target"])
prediction.to_csv("data/submission.csv")
print("Done\n")

# Scaling data

scaler = preprocessing.StandardScaler().fit(X_train[:,bin_cat_vars])

X_train[:,bin_cat_vars] = scaler.transform(X_train[:,bin_cat_vars])
X_validate[:,bin_cat_vars] = scaler.transform(X_validate[:,bin_cat_vars])
X_test[:,bin_cat_vars] = scaler.transform(X_test[:,bin_cat_vars])


