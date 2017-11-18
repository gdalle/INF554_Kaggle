import pandas as pd
import numpy as np
from sklearn import linear_model,ensemble
from evaluation import *
from data_processing import *
from sklearn import preprocessing, model_selection,pipeline,metrics,externals

# Read csv files

print("=================")
print("Loading data ...")

train = pd.read_csv("data/train.csv", index_col=0)
test = pd.read_csv("data/test.csv", index_col=0)

train_features = [
    "ps_car_13",             
	"ps_reg_03",         
	"ps_ind_05_cat", 
	"ps_ind_03", 
	"ps_ind_15", 
	"ps_reg_02", 
	"ps_car_14", 
	"ps_car_12", 
	"ps_car_01_cat",  
	"ps_car_07_cat", 
	"ps_ind_17_bin", 
	"ps_car_03_cat", 
	"ps_reg_01", 
	"ps_car_15", 
	"ps_ind_01",  
	"ps_ind_16_bin", 
	"ps_ind_07_bin",  
	"ps_car_06_cat", 
	"ps_car_04_cat",  
	"ps_ind_06_bin", 
	"ps_car_09_cat",  
	"ps_car_02_cat",  
	"ps_ind_02_cat", 
	"ps_car_11",
	"ps_car_05_cat",  
	"ps_ind_08_bin",  
	"ps_car_08_cat", 
	"ps_ind_09_bin",  
	"ps_ind_04_cat",  
	"ps_ind_18_bin",
	"ps_ind_12_bin",
	"ps_ind_14",
	"target",
]

train = train[train_features]
test = test[train_features[:-1]]

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
    ("binarizer", preprocessing.OneHotEncoder())
])

preprocessor = pipeline.FeatureUnion([("num", num_pipeline), ("bin", bin_pipeline), ("cat", cat_pipeline)])

base = ensemble.GradientBoostingClassifier(verbose=2)
pipe = pipeline.Pipeline([
    ("preprocessor", preprocessor),
    ("clf",base)
])

cv = model_selection.StratifiedKFold(n_splits=4)
scorer = metrics.make_scorer(gini_scorer,needs_proba=True)
print("Done\n")


print("=================")
print("Grid searching over hyperparameters ...")
param_grid={'clf__max_depth': [3,4,5,6,7,8,9,10]}
gs = model_selection.GridSearchCV(pipe,param_grid=param_grid,scoring=scorer,cv=cv,n_jobs=4,verbose=5)
gs.fit(X0,y0)

print("=================")
print("Predicting for submission ...")
y_test_p = gs.predict_proba(X_test)[:,1]
prediction = pd.DataFrame(
    index=test.index,
    data=np.round(y_test_p, 3),
    columns=["target"])
prediction.to_csv("output/submission.csv")
print("Done\n")

externals.joblib.dump(gs,"GB_GS_max_depth.pkl")