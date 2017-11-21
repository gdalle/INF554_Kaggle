"""Creating features."""

import pandas as pd
import numpy as np
import extraction as ex


def count_days(date_series, dataset="train", base_date=None):
    """
    Count the days elapsed since the base date for a series of dates.
    
    The base date depends on whether we consider
    the train set (predict churn for February 2017)
    or the test set (March 2017).
    """
    if base_date is None:
        if dataset == "train":
            base_date = pd.Timestamp(2017, 3, 1)
        elif dataset == "test":
            base_date = pd.Timestamp(2017, 4, 1)

    if type(date_series.iloc[0]) == type(base_date):
        # We are dealing with dates : create delta
        td_ns = pd.to_timedelta(date_series - base_date).values
    else:
        # We are already dealing with deltas
        td_ns = date_series.values
    # The default time deltas are in nanoseconds, we want them in days
    td_days = (td_ns / (np.power(10, 9) * 3600 * 24)).astype(int)
    return td_days


def categorize(df, col):
    """Turn categorical variables into 0/1 columns for each possible value."""
    cat = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, cat], axis=1)
    df = df.drop(col, axis=1)
    return df


def get_useful_users(
    users_table, members=None, transactions=None, user_logs=None
):
    """Get subset from train or test whose users appear in other tables."""
    print("Getting useful users")
    useful_indices = np.ones((len(users_table.index))).astype(bool)
    if members is not None:
        print("- Members")
        useful_indices = useful_indices & \
            users_table.index.isin(members.index)
    if transactions is not None:
        print("- Transactions")
        useful_indices = useful_indices & \
            users_table.index.isin(transactions["msno"].unique())
    if user_logs is not None:
        print("- User logs")
        useful_indices = useful_indices & \
            users_table.index.isin(user_logs["msno"].unique().astype(str))
    return users_table[useful_indices]


def select_features(train_full, test_full, features):
    """Select features to use in the learning phase."""
    features = list(features)
    features_train = features + ["is_churn"]
    train_filtered = train_full.loc[:, features_train]
    test_filtered = test_full.loc[:, features]
    return train_filtered, test_filtered


def normalize_features(train_filtered, test_filtered):
    """Standardize features with values outside of [0, 1]."""
    for c in test_filtered.columns:
        # If not binary
        if train_filtered[c].max() > 1.001 or train_filtered[c].min() < -0.001:
            # Standardize in train
            # Standardize in test with the scale parameters of train
            m = train_filtered[c].mean()
            s = train_filtered[c].std()
            train_filtered[c] = (train_filtered[c] - m) / s
            test_filtered[c] = (test_filtered[c] - m) / s
    return train_filtered, test_filtered


# TEST

if __name__ == "__main__":
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
    train_useful = get_useful_users(train, members=members, transactions=transactions, user_logs=user_logs)
    test_useful = get_useful_users(test, members=members, transactions=transactions, user_logs=user_logs)

    # Exploit the tables
    members_data = exploit_members(members)
    transactions_data = exploit_transactions(transactions)
    user_logs_data = exploit_user_logs(user_logs)

    data_list = [members_data, transactions_data, user_logs_data]

    # Add the data to the train set and test set
    train_full = add_data_to_users(train_useful, data_list)
    test_full = add_data_to_users(test_useful, data_list)

    # Keep only the features we want
    features = test_full.columns # all of them
    train_filtered, test_filtered = select_features(train_full, test_full, features)

    # Normalize the columns
    train_filtered, test_filtered = normalize_features(train_filtered, test_filtered)
