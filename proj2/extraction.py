"""Reading data from big tables."""

import pandas as pd
import numpy as np
import itertools
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss, make_scorer
from datetime import datetime

global_path = "/tmp/kaggle/proj2/"
# global_path = ""


def memory(df):
    """Print memory usage of a DataFrame."""
    mem = df.memory_usage() / (1024**2)
    print("\nMemory usage (MB) :", mem.sum())
    print(mem)
    print("")


# Train table

def read_train():
    """Read train."""
    print("\nREADING TRAIN\n")
    train = pd.read_csv(global_path + "data/train_v2.csv")

    # Change integer storage to gain memory
    train["is_churn"] = train["is_churn"].astype(np.int8)

    # Change index
    train.index = train["msno"]
    train = train.drop("msno", axis=1)

    memory(train)

    return train


# Test table

def read_test():
    """Read test."""
    print("\nREADING TEST\n")
    test = pd.read_csv(global_path + "data/sample_submission_v2.csv")

    # Lose the bogus target column
    test = test.drop("is_churn", axis=1)

    # Change index
    test.index = test["msno"]
    test = test.drop("msno", axis=1)

    memory(test)

    return test


# Members table

def read_members(split_dates=False):
    """Read members."""
    print("\nREADING MEMBERS\n")
    dtype_cols_members = {
        'msno': object,
        'city': np.int8,
        'bd': np.int16,
        'gender': object,
        'registered_via': np.int8,
        'expiration_date': object,
        'registration_init_time': object,
    }

    members = pd.read_csv(
        global_path + "data/members_v3.csv", dtype=dtype_cols_members)

    # Recode genrer as integers
    members["gender"] = members["gender"].replace("male", 1)
    members["gender"] = members["gender"].replace("female", 2)
    members["gender"] = members["gender"].replace(np.NaN, 0)
    members["gender"] = members["gender"].astype(np.int8)

    if not split_dates:
        members["registration_init_time"] = \
            pd.to_datetime(members["registration_init_time"])

    if split_dates:
        # Split date on three columns
        t = pd.to_datetime(members["registration_init_time"])
        members['registration_init_year'] = t.dt.year.astype(np.int16)
        members['registration_init_month'] = t.dt.month.astype(np.int8)
        members['registration_init_day'] = t.dt.day.astype(np.int8)
        members = members.drop("registration_init_time", axis=1)

    # Change index
    members.index = members["msno"]
    members = members.drop("msno", axis=1)

    memory(members)

    return members


# Transactions table

def read_transactions(
    useful_msno=None,
    max_lines=np.inf, chunksize=10**5,
    split_dates=False
):
    """Read transactions."""
    print("\nREADING TRANSACTIONS\n")

    dtype_cols_transactions = {
        'msno': object,
        'payment_method_id': np.int8,
        'payment_plan_days': np.int16,
        'plan_list_price': np.int16,
        'actual_amount_paid': np.int16,
        'is_auto_renew': np.int8,
        'transaction_date': object,
        'membership_expire_date': object,
        'is_cancel': np.int8
    }

    iterator1 = pd.read_csv(
        global_path + "data/transactions.csv",
        chunksize=chunksize,
        iterator=True,
        header=0,
        dtype=dtype_cols_transactions
    )
    iterator2 = pd.read_csv(
        global_path + "data/transactions_v2.csv",
        chunksize=chunksize,
        iterator=True,
        header=0,
        dtype=dtype_cols_transactions
    )

    transactions_list = []

    # Read data by chunks to alleviate memory load
    chunk_number = 0
    for df in itertools.chain(iterator1, iterator2):
        if useful_msno is not None:
            append_condition = df['msno'].isin(useful_msno)
            df = df[append_condition]
        transactions_list.append(df)
        print("Chunk {} of transactions read".format(chunk_number + 1))
        chunk_number += 1
        if chunk_number >= max_lines / chunksize:
            break

    transactions = pd.concat(transactions_list, ignore_index=True)

    # Change integer storage
    transactions.index = transactions.index.astype(np.int32)

    if not split_dates:
        transactions["transaction_date"] = \
            pd.to_datetime(transactions["transaction_date"])
        transactions["membership_expire_date"] = \
            pd.to_datetime(transactions["membership_expire_date"])

    if split_dates:

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

    return transactions


# User logs table

def read_user_logs(
    useful_msno=None, just_date=False,
    max_lines=np.inf, chunksize=10**5,
    split_dates=False
):
    """Read user logs."""
    print("\nREADING USER LOGS\n")

    dtype_cols_user_logs = {
        'msno': object,
        'date': np.int64,
        'num_25': np.int8,
        'num_50': np.int8,
        'num_75': np.int8,
        'num_985': np.int8,
        'num_100': np.int8,
        'num_unq': np.int8,
        'total_secs': np.float32
    }

    if not just_date:
        # Read all columns
        iterator1 = pd.read_csv(
            global_path + "data/user_logs.csv",
            chunksize=chunksize,
            iterator=True,
            header=0,
            dtype=dtype_cols_user_logs,
            usecols=["date", "msno", "num_100", "num_unq", "total_secs"]
        )
        iterator2 = pd.read_csv(
            global_path + "data/user_logs_v2.csv",
            chunksize=chunksize,
            iterator=True,
            header=0,
            dtype=dtype_cols_user_logs,
            usecols=["date", "msno", "num_100", "num_unq", "total_secs"]
        )

    if just_date:
        # Read just date and user id
        iterator1 = pd.read_csv(
            global_path + "data/user_logs.csv",
            chunksize=chunksize,
            iterator=True,
            header=0,
            dtype=dtype_cols_user_logs,
            usecols=["date", "msno"]
        )
        iterator2 = pd.read_csv(
            global_path + "data/user_logs_v2.csv",
            chunksize=chunksize,
            iterator=True,
            header=0,
            dtype=dtype_cols_user_logs,
            usecols=["date", "msno"]
        )

    user_logs_list = []

    # Read data by chunks to alleviate memory load
    chunk_number = 0
    for df in itertools.chain(iterator1, iterator2):
        if useful_msno is not None:
            append_condition = df['msno'].isin(useful_msno)
            df = df[append_condition]
        df["date"] = pd.to_datetime(df["date"].astype(str))
        user_logs_list.append(df)
        print("Chunk {} of user logs read".format(chunk_number + 1))
        chunk_number += 1
        if chunk_number >= max_lines / chunksize:
            break

    user_logs = pd.concat(user_logs_list, ignore_index=True)

    user_logs['msno'] = user_logs['msno'].astype('category')

    # Change integer storage
    for col in ['num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq']:
        if col in user_logs.columns:
            pass
            # user_logs[col] = user_logs[col].astype(np.int8)

    if split_dates:
        # Split date on three columns
        t3 = user_logs["date"]
        user_logs['year'] = t3.dt.year.astype(np.int16)
        user_logs['month'] = t3.dt.month.astype(np.int8)
        user_logs['day'] = t3.dt.day.astype(np.int8)
        user_logs = user_logs.drop("date", axis=1)

    memory(user_logs)

    return user_logs


# TEST

if __name__ == "__main__":
    train = read_train()
    test = read_test()
    members = read_members()
    useful_msno = set.union(
        set(train.index.unique()),
        set(test.index.unique())
    )
    transactions = read_transactions(useful_msno=useful_msno, max_lines=10**6 // 3)
    user_logs = read_user_logs(useful_msno=useful_msno, just_date=True, max_lines=10**6 // 3)
