"""Reading data and creating output."""

import pandas as pd
import numpy as np
from sklearn import linear_model
from datetime import datetime


def memory(df):
    """Print memory usage of a DataFrame."""
    mem = df.memory_usage() / (1024**2)
    print("\nMemory usage (MB) :", mem.sum())
    print(mem)
    print("")


def categorize(df, col):
    """Turn categorical variables into 0/1 columns for each possible value."""
    cat = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, cat], axis=1)
    df = df.drop(col, axis=1)
    return df


# Train table

def read_train():
    """Read train."""
    print("\nREADING TRAIN\n")
    train2 = pd.read_csv("data/train_v2.csv")
    # train1 = pd.read_csv("data/train.csv")
    # train2 = train2[~train2.msno.isin(train1.msno.unique())]
    # train = pd.merge(train1, train2, how="outer")
    # train.index = train["msno"]
    # train = train.drop("msno", axis=1)
    # del train1
    # del train2

    train = train2

    # Change integer storage to gain memory
    train["is_churn"] = train["is_churn"].astype(np.int8)
    memory(train)

    return train


# Test table

def read_test():
    """Read test."""
    print("\nREADING TEST\n")
    test2 = pd.read_csv("data/sample_submission_v2.csv")
    # test1 = pd.read_csv("data/sample_submission_zero.csv")
    # test2 = test2[~test2.msno.isin(test1.msno.unique())]
    # test = pd.merge(test1, test2, how="outer")
    # test.index = test["msno"]
    # test = test.drop("msno", axis=1)
    # del test1
    # del test2

    test = test2

    # Lose the bogus target column
    test = test.drop("is_churn", axis=1)
    memory(test)

    return test


# Members table

def read_members():
    """Read members."""
    print("\nREADING MEMBERS\n")
    dtype_cols_members = {
        'msno': object,
        'city': np.int64,
        'bd': np.int64,
        'gender': object,
        'registered_via': np.int64,
        'expiration_date': object,
        'registration_init_time': object,
    }

    members1 = pd.read_csv("data/members.csv", dtype=dtype_cols_members)
    members2 = pd.read_csv("data/members_v2.csv", dtype=dtype_cols_members)
    members2 = members2[~members2.msno.isin(members1.msno.unique())]
    members = pd.merge(members1, members2, how="outer")

    # Recode genrer as integers
    members["gender"] = members["gender"].replace("male", 1)
    members["gender"] = members["gender"].replace("female", 2)
    members["gender"] = members["gender"].replace(np.NaN, 0)
    members["gender"] = members["gender"].astype(np.int8)

    # Change integer storage
    members['city'] = members['city'].astype(np.int8)
    members['bd'] = members['bd'].astype(np.int16)
    members['registered_via'] = members['registered_via'].astype(np.int8)

    # Drop expiration date, which isn't available for everyone
    members = members.drop("expiration_date", axis=1)

    # Split date on three columns
    t = pd.to_datetime(members["registration_init_time"])
    members['registration_init_year'] = t.dt.year.astype(np.int16)
    members['registration_init_month'] = t.dt.month.astype(np.int8)
    members['registration_init_day'] = t.dt.day.astype(np.int8)
    members = members.drop("registration_init_time", axis=1)

    memory(members)

    del members1
    del members2
    return members


# Transactions table

def read_transactions(train, test, max_lines=np.inf):
    """Read transactions."""
    print("\nREADING TRANSACTIONS\n")
    # Useful ids
    id_train = set(train.index.unique())
    id_test = set(test.index.unique())
    useful_msno = set.union(id_train, id_test)

    dtype_cols_transactions = {
        'msno': object,
        'payment_method_id': np.int64,
        'payment_plan_days': np.int64,
        'plan_list_price': np.int64,
        'actual_amount_paid': np.int64,
        'is_auto_renew': np.int64,
        'transaction_date': object,
        'membership_expire_date': object,
        'is_cancel': np.int64
    }

    transactions_list1 = []
    transactions_list2 = []

    # Read data by chunks to alleviate memory load
    chunk_number = 0
    for df in pd.read_csv(
        "data/transactions.csv",
        chunksize=10**5,
        iterator=True,
        header=0,
        dtype=dtype_cols_transactions
    ):
        append_condition = df['msno'].isin(useful_msno)
        df = df[append_condition]
        transactions_list1.append(df)
        print("Chunk {} of table 1 read".format(chunk_number))
        chunk_number += 1
        if chunk_number >= max_lines / (10**5):
            break

    transactions1 = pd.concat(transactions_list1, ignore_index=True)

    # Read data by chunks to alleviate memory load
    chunk_number = 0
    for df in pd.read_csv(
        "data/transactions_v2.csv",
        chunksize=10**5,
        iterator=True,
        header=0,
        dtype=dtype_cols_transactions
    ):
        append_condition = df['msno'].isin(useful_msno)
        df = df[append_condition]
        transactions_list2.append(df)
        print("Chunk {} of table 2 read".format(chunk_number))
        chunk_number += 1
        if chunk_number >= max_lines / (10**5):
            break

    transactions2 = pd.concat(transactions_list2, ignore_index=True)

    # transactions1 = pd.read_csv(
    #     "data/transactions.csv", dtype=dtype_cols_transactions)
    # transactions2 = pd.read_csv(
    #     "data/transactions_v2.csv", dtype=dtype_cols_transactions)

    transactions2 = \
        transactions2[~transactions2.msno.isin(transactions1.msno.unique())]

    transactions = pd.merge(
        transactions1, transactions2,
        how="outer", sort=False)

    # Change integer storage
    transactions.index = transactions.index.astype(np.int32)
    transactions['payment_method_id'] = \
        transactions['payment_method_id'].astype(np.int8)
    transactions['payment_plan_days'] = \
        transactions['payment_plan_days'].astype(np.int16)
    transactions['plan_list_price'] = \
        transactions['plan_list_price'].astype(np.int16)
    transactions['actual_amount_paid'] = \
        transactions['actual_amount_paid'].astype(np.int16)
    transactions['is_auto_renew'] = \
        transactions['is_auto_renew'].astype(np.int8)
    transactions['is_cancel'] = \
        transactions['is_cancel'].astype(np.int8)

    # Split date on three columns
    t1 = pd.to_datetime(transactions["transaction_date"])
    transactions['transaction_year'] = t1.dt.year.astype(np.int16)
    transactions['transaction_month'] = t1.dt.month.astype(np.int8)
    transactions['transaction_day'] = t1.dt.day.astype(np.int8)
    transactions = transactions.drop("transaction_date", axis=1)

    # Split date on three columns
    t2 = pd.to_datetime(transactions["membership_expire_date"])
    transactions['membership_expire_year'] = t2.dt.year.astype(np.int16)
    transactions['membership_expire_month'] = t2.dt.month.astype(np.int8)
    transactions['membership_expire_day'] = t2.dt.day.astype(np.int8)
    transactions = transactions.drop("membership_expire_date", axis=1)

    memory(transactions)

    del transactions1
    del transactions2
    return transactions


# User logs table

def read_user_logs(train, test, max_lines=np.inf):
    """Read user logs."""
    print("\nREADING USER LOGS\n")
    # Useful ids
    id_train = set(train.index.unique())
    id_test = set(test.index.unique())
    useful_msno = set.union(id_train, id_test)

    dtype_cols_user_logs = {
        'msno': object,
        'date': np.int64,
        'num_25': np.int32,
        'num_50': np.int32,
        'num_75': np.int32,
        'num_985': np.int32,
        'num_100': np.int32,
        'num_unq': np.int32,
        'total_secs': np.float32
    }

    user_logs_list1 = []
    user_logs_list2 = []

    # Read data by chunks to alleviate memory load
    chunk_number = 0
    for df in pd.read_csv(
        "data/user_logs.csv",
        chunksize=10**5,
        iterator=True,
        header=0,
        dtype=dtype_cols_user_logs
    ):
        append_condition = df['msno'].isin(useful_msno)
        df = df[append_condition]
        df["date"] = pd.to_datetime(df["date"])
        user_logs_list1.append(df)
        print("Chunk {} of table 1 read".format(chunk_number))
        chunk_number += 1
        if chunk_number >= max_lines / (10**5):
            break

    user_logs1 = pd.concat(user_logs_list1, ignore_index=True)

    # Read data by chunks to alleviate memory load
    chunk_number = 0
    for df in pd.read_csv(
        "data/user_logs_v2.csv",
        chunksize=10**5,
        iterator=True,
        header=0,
        dtype=dtype_cols_user_logs
    ):
        append_condition = df['msno'].isin(useful_msno)
        df = df[append_condition]
        df["date"] = pd.to_datetime(df["date"])
        user_logs_list2.append(df)
        print("Chunk {} of table 2 read".format(chunk_number))
        chunk_number += 1
        if chunk_number >= max_lines / (10**5):
            break

    user_logs2 = pd.concat(user_logs_list2, ignore_index=True)

    user_logs = pd.merge(
        user_logs1, user_logs2,
        how="outer", sort=False, copy=False)

    user_logs['msno'] = user_logs['msno'].astype('category')

    # Change integer storage
    for col in ['num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq']:
        user_logs[col] = user_logs[col].astype(np.int8)

    # Split date on three columns
    t3 = user_logs["date"]
    user_logs['year'] = t3.dt.year.astype(np.int16)
    user_logs['month'] = t3.dt.month.astype(np.int8)
    user_logs['day'] = t3.dt.day.astype(np.int8)
    user_logs = user_logs.drop("date", axis=1)

    memory(user_logs)

    del user_logs1
    del user_logs2
    return user_logs


# MAIN

# User dataframes

train = read_train()
test = read_test()
members = read_members()

train.index = train["msno"]
train = train.drop("msno", axis=1)
test.index = test["msno"]
test = test.drop("msno", axis=1)
members.index = members["msno"]
members = members.drop("msno", axis=1)

# Transactions and user logs dataframes

transactions = read_transactions(train, test, max_lines=10**6)
user_logs = read_user_logs(train, test, max_lines=10**6)

# Feature extraction

print("\nFEATURE EXTRACTION\n")

# Users on which we have info
train_useful = train[
    train.index.isin(members.index) &
    train.index.isin(transactions["msno"]) &
    train.index.isin(user_logs["msno"].astype(object))
].copy()
test_useful = test.loc[
    test.index.isin(members.index) &
    test.index.isin(transactions["msno"]) &
    test.index.isin(user_logs["msno"].astype(object))
].copy()


def add_members_info(user_table, members):
    """Extract info from members and add it to train or test."""
    user_table["registration_init_year"] = \
        members.loc[user_table.index, ["registration_init_year"]]
    user_table["city"] = members.loc[user_table.index, ["city"]]
    user_table = categorize(user_table, "city")
    return user_table


# Add features directly taken from members table
train_useful = add_members_info(train_useful, members)
test_useful = add_members_info(test_useful, members)

# Prediction

print("\nPREDICTION\n")

# Conversion into arrays for scikit-learn
x = np.array(train_useful.drop("is_churn", axis=1))
y = np.array(train_useful["is_churn"])
xt = np.array(test_useful)

# Linear Regression
clf = linear_model.LinearRegression()
clf.fit(x, y)
yt = clf.predict(xt)
a = yt < 0
b = yt > 1
c = a | b
yt[c] = 0

# Random prediction as baseline
test["is_churn"] = np.random.rand(len(test)) * 0.1
# For users on which we have more info, use it
test.loc[test_useful.index, ["is_churn"]] = yt.reshape(-1, 1)

# Save as csv
submission = test.loc[:, ["is_churn"]]
submission.to_csv("data/submission.csv")
