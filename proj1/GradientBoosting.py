import pandas as pd
import numpy as np
from sklearn import linear_model,ensemble
from evaluation import *
from data_processing import *
from sklearn import preprocessing, model_selection,pipeline,metrics,externals,feature_extraction

# Read csv files

print("=================")
print("Loading data ...")

train = pd.read_csv("/tmp/data/train.csv", index_col=0)
test = pd.read_csv("/tmp/data/test.csv", index_col=0)

print("Done\n")

print("=================")
print("Processing data ...")

# Turn DataFrames into arrays for scikit-learn
# Forget the target column in X
train_0 = train.drop("target", axis=1)

name_to_index = {name: train_0.columns.get_loc(name) for name in train_0.columns}

X0 = np.array(train_0)
y0 = np.array(train["target"])

X_test = np.array(test)

print("Done\n")

print("=================")
print("Setting the pipeline ...")

# Define the num pipeline
num_selector = filter_num_transform(name_to_index)
num_imputer = preprocessing.Imputer(missing_values=-1, strategy="mean")
num_pipeline = pipeline.Pipeline([
    ("selector", num_selector),
    ("imputer", num_imputer),
    ("scaler", preprocessing.StandardScaler())
])

# Define the bin pipeline
bin_selector = filter_bin_transform(name_to_index)
bin_imputer = preprocessing.Imputer(missing_values=-1, strategy="most_frequent")
bin_pipeline = pipeline.Pipeline([
    ("selector", bin_selector),
    ("imputer", bin_imputer)
])

# Define the cat pipeline
cat_selector = filter_cat_transform(name_to_index)
cat_imputer = preprocessing.Imputer(missing_values=-1, strategy="most_frequent")
cat_pipeline = pipeline.Pipeline([
    ("selector", cat_selector),
    ("imputer", cat_imputer),
#   ("binarizer", preprocessing.OneHotEncoder())
])

preprocessor = pipeline.FeatureUnion([("num", num_pipeline), ("bin", bin_pipeline), ("cat", cat_pipeline)])

base = ensemble.GradientBoostingClassifier(verbose=5,subsample=.8,
                                           max_features = "sqrt",max_depth=8,
                                           min_samples_leaf = .01,
                                           n_estimators = 1500, learning_rate = .05)

pipe = pipeline.Pipeline([
    ("preprocessor", preprocessor),
    ("clf",base)
])

print("Done\n")

print("=================")
print("Cross-validation ...")
cv = model_selection.StratifiedKFold(n_splits=4)
scorer = metrics.make_scorer(gini_scorer,needs_proba=True)
cross_val = model_selection.cross_val_score(pipe,X0,y0,cv=cv,scoring=scorer,verbose=5,n_jobs=4)
print(cross_val)
print("Done\n")

print("=================")
print("Learning on the whole training set ...")
pipe.fit(X0,y0)

print("=================")
print("Predicting for submission ...")
y_test_p = pipe.predict_proba(X_test)[:,1]
prediction = pd.DataFrame(
    index=test.index,
    data=np.round(y_test_p, 3),
    columns=["target"])
prediction.to_csv("output/submission.csv")
print("Done\n")

externals.joblib.dump(pipe,"GradientBoosting_well_tuned.pkl")

