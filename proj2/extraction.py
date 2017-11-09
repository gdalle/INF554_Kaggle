"""Reading data and creating output."""

import pandas as pd
import numpy as np
from sklearn import linear_model
from time import time


def memory(df):
    """Print memory usage."""
    mem = df.memory_usage() / (1024**2)
    print("Memory usage (MB) :", mem.sum())
    print(mem)


def to_day(date, start_year=2015):
    """Count the number of days since 01/01/start_year."""
    return date.dayofyear + 365 * (date.year - start_year)


# Train table

def read_train():
    """Read train."""
    train1 = pd.read_csv("data/train.csv")
    train2 = pd.read_csv("data/train_v2.csv")
    train2 = train2[~train2.msno.isin(train1.msno.unique())]
    train = pd.merge(train1, train2, how="outer")
    # train.index = train["msno"]
    # train = train.drop("msno", axis=1)

    memory(train)
    train["is_churn"] = train["is_churn"].astype(np.int8)
    memory(train)

    del train1
    del train2
    return train


train = read_train()


# Test table

def read_test():
    """Read test."""
    test2 = pd.read_csv("data/sample_submission_v2.csv")
    # test1 = pd.read_csv("data/sample_submission_zero.csv")
    # test2 = test2[~test2.msno.isin(test1.msno.unique())]
    # test = pd.merge(test1, test2, how="outer")
    # test.index = test["msno"]
    # test = test.drop("msno", axis=1)

    test = test2
    memory(test)
    test = test.drop("is_churn", axis=1)
    memory(test)

    # del test1
    # del test2

    return test


test = read_test()


# Members table

def read_members():
    """Read members."""
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

    memory(members)

    # members.index = members["msno"]
    # members = members.drop("msno", axis=1)

    members["gender"] = members["gender"].replace("male", 1)
    members["gender"] = members["gender"].replace("female", 2)
    members["gender"] = members["gender"].replace(np.NaN, 0)
    members["gender"] = members["gender"].astype(np.int8)

    members['city'] = members['city'].astype(np.int8)
    members['bd'] = members['bd'].astype(np.int16)
    members['registered_via'] = members['registered_via'].astype(np.int8)

    members = members.drop("expiration_date", axis=1)
    t = pd.to_datetime(members["registration_init_time"])
    members['registration_init_year'] = t.dt.year.astype(np.int16)
    members['registration_init_month'] = t.dt.month.astype(np.int8)
    members['registration_init_day'] = t.dt.day.astype(np.int8)
    members = members.drop("registration_init_time", axis=1)

    memory(members)

    del members1
    del members2
    return members


members = read_members()


# Transactions table

def read_transactions():
    """Read transactions."""
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

    transactions1 = pd.read_csv(
        "data/transactions.csv", dtype=dtype_cols_transactions)
    transactions2 = pd.read_csv(
        "data/transactions_v2.csv", dtype=dtype_cols_transactions)
    transactions2 = \
        transactions2[~transactions2.msno.isin(transactions1.msno.unique())]
    transactions = pd.merge(transactions1, transactions2, how="outer")

    memory(transactions)
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

    t1 = pd.to_datetime(transactions["transaction_date"])
    transactions['transaction_year'] = t1.dt.year.astype(np.int16)
    transactions['transaction_month'] = t1.dt.month.astype(np.int8)
    transactions['transaction_day'] = t1.dt.day.astype(np.int8)
    transactions = transactions.drop("transaction_date", axis=1)

    t2 = pd.to_datetime(transactions["membership_expire_date"])
    transactions['membership_expire_year'] = t2.dt.year.astype(np.int16)
    transactions['membership_expire_month'] = t2.dt.month.astype(np.int8)
    transactions['membership_expire_day'] = t2.dt.day.astype(np.int8)
    transactions = transactions.drop("membership_expire_date", axis=1)

    memory(transactions)

    del transactions1
    del transactions2
    return transactions


transactions = read_transactions()


# User logs table

def read_user_logs(train, test, max_lines=4*(10**7)):
    """Read user logs."""
    # Useful ids
    id_train = set(train["msno"].unique())
    id_test = set(test["msno"].unique())
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

    user_logs = None
    chunk_number = 0
    user_logs_list = []
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
        user_logs_list.append(df)
        print("Chunk {} read".format(chunk_number))
        chunk_number += 1
        if chunk_number >= max_lines / (10**5):
            break
    user_logs = pd.concat(user_logs_list, ignore_index=True)

    user_logs['msno'] = user_logs['msno'].astype('category')

    memory(user_logs)
    return user_logs


user_logs = read_user_logs(train, test, max_lines=100000)


# Merging train / test with members as a way of getting first features

train = pd.merge(train, members, how='left', on='msno')
train.index = train["msno"]
train = train.drop("msno", axis=1)
test = pd.merge(test, members, how='left', on='msno')
test.index = test["msno"]
test = test.drop("msno", axis=1)

test["is_churn"] = np.random.rand(len(test))
submission = test.loc[:, ["is_churn"]]
len(submission)
submission.to_csv("data/submission.csv")
