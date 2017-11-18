import pandas as pd

def filter_transform(col_names, name_to_index):
    def result(X):
        l = [name_to_index[col] for col in col_names]
        return X[:, l]
    return result

def filter_num_transform(name_to_index):
    return filterTransform([name for name in name_to_index.keys() if ("cat" not in name and "bin" not in name)], name_to_index)

def filter_cat_transform(name_to_index):
    return filterTransform([name for name in name_to_index.keys() if "cat" in name], name_to_index)

def filter_bin_transform(name_to_index):
    return filterTransform([name for name in name_to_index.keys() if "bin" in name], name_to_index)

class filterTransform(object):
    def __init__(self, col_names, name_to_index):
        self.idx = [name_to_index[col] for col in col_names]
        
    def transform(self, X):
        X0 = X.copy()
        return X0[:, self.idx]

    def fit(self, X, y=None):
        return self