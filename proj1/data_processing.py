import pandas as pd

def is_cat_bin(data):

    """ Return a list of the indexes of the binary / categorical columns"""

    res = []
    i = 0
    for c in data.columns:
        if "bin" in c or "cat" in c:
            res.append(i)
        i+=1
    return res

def binarize_cat(data):
    """Return a new DataFrame with categorical variables encoded as 0/1."""
    for c in data.columns:
        if "cat" in c:
            # Get 0/1 columns associated with categories
            dummies = pd.get_dummies(data[c])
            # Change column names to form var==1, var==2...
            dummies.columns = [c + "=" + str(dc) for dc in dummies.columns]
            # Concatenate to original data
            data = data.drop(c, axis=1).join(dummies)
    return data


def filter_shitty_columns(data, max_missing=1, display=False):
    """Find out which columns have too much missing data."""
    total = len(data)
    shitty_columns = []
    for c in data.columns:
        count = data[c].value_counts()
        # Compute number of missing entries
        if -1 in count.index:
            missing = count.loc[-1]
            missing_percent = np.round(missing / total * 100, 3)
        else:
            missing = 0
            missing_percent = 0
        if display:
            # Print info
            print()
            print("=================")
            print(c)
            print("{} different values".format(len(data[c].unique())))
            print("{} missing entries, i.e. {} %".format(
                missing, missing_percent))
            print(count.head())
            print("=================")
        # If the percentage of missing data is too high, keep track
        if missing_percent > max_missing:
            shitty_columns.append(c)

    for sc in shitty_columns:
        data = data.drop(sc, axis=1)
    return data

