import pandas as pd

def filter_num_transform(name_to_index):
    return filterTransform([name for name in name_to_index.keys() if ("cat" not in name and "bin" not in name)],
                           name_to_index)


def filter_cat_transform(name_to_index):
    return filterTransform([name for name in name_to_index.keys() if "cat" in name], name_to_index)


def filter_bin_transform(name_to_index):
    return filterTransform([name for name in name_to_index.keys() if "bin" in name], name_to_index)


class filterTransform(object):
    def __init__(self, col_names, name_to_index):
        self.name_to_index = name_to_index
        self.col_names = col_names
        self.idx = [name_to_index[col] for col in col_names]

    def transform(self, X):
        X0 = X.copy()
        return X0[:, self.idx]

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=None):
        return {"col_names": self.col_names, "name_to_index": self.name_to_index}