"""Creating features."""

import pandas as pd
import numpy as np
import extraction as ex


def count_days(date_series, base_date=pd.Timestamp(2000, 1, 1)):
    """Count the days elapsed since the base date for a series of dates."""
    td = pd.to_timedelta(date_series - base_date)
    return td.apply(lambda x: x.days)


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
            users_table.index.isin(transactions["msno"])
    if user_logs is not None:
        print("- User logs")
        useful_indices = useful_indices & \
            users_table.index.isin(user_logs["msno"].astype(str))
    return users_table[useful_indices]


def add_members_info(users_table, members):
    """
    Extract info from members and add it to train or test.

    We assume the users in users_table are all present in the members.
    """
    print("Adding members info")
    useful_members = members.loc[users_table.index, :]

    # print("- Categories")
    # # City (categories)
    # users_table = users_table.assign(
    #     city=useful_members.loc[:, ["city"]])
    # users_table = categorize(users_table, "city")
    # # Gender (categories)
    # users_table = users_table.assign(
    #     gender=useful_members.loc[:, ["gender"]])
    # users_table = categorize(users_table, "gender")
    # # Registered via (categories)
    # users_table = users_table.assign(
    #     registered_via=useful_members.loc[:, ["registered_via"]])
    # users_table = categorize(users_table, "registered_via")
    # Registration (measured in days since 01/01/2000) TODO
    print("- Registration init time")
    t = useful_members["registration_init_time"]
    users_table = users_table.assign(
        registration_init_time=count_days(t))
    return users_table


def add_transactions_info(users_table, transactions):
    """
    Extract info from transactions and add it to train or test.

    We assume the users in users_table are all present in the user_logs.
    """
    print("Adding transactions info")
    grouped_trans = transactions.groupby("msno")
    print("- Latest transactions")
    max_trans = grouped_trans.max()
    max_trans_useful = max_trans.loc[users_table.index, :]
    max_trans_useful = max_trans_useful.loc[
        :, ["transaction_date", "membership_expire_date"]
    ]
    max_trans_useful = max_trans_useful.apply(count_days)
    users_table = pd.concat([users_table, max_trans_useful], axis=1)
    return users_table


def add_user_logs_info(users_table, user_logs):
    """
    Extract info from user_logs and add it to train or test.

    We assume the users in users_table are all present in the user_logs.
    """
    print("Adding user logs info")
    print("- Grouping")
    grouped_logs = user_logs.drop("date", axis=1).groupby("msno")
    print("- Mean logs")
    mean_logs = grouped_logs.mean()
    mean_logs_useful = mean_logs.loc[users_table.index, :]
    users_table = pd.concat([users_table, mean_logs_useful], axis=1)
    return users_table


def add_all_info(users_table, members=None, transactions=None, user_logs=None):
    """Extract info from all other tables and add it to train or test."""
    if members is not None:
        users_table = add_members_info(users_table, members)
    if transactions is not None:
        users_table = add_transactions_info(users_table, transactions)
    if user_logs is not None:
        users_table = add_user_logs_info(users_table, user_logs)
    return users_table


# TEST

if __name__ == "__main__":
    # Reading tables
    train = ex.read_train()
    test = ex.read_test()
    members = ex.read_members()
    transactions = ex.read_transactions(train, test, max_lines=10**6 // 3)
    user_logs = ex.read_user_logs(train, test, max_lines=10**6 // 3)

    # Get useful users
    train_useful = get_useful_users(
        train,
        members=members, transactions=transactions, user_logs=user_logs)
    test_useful = get_useful_users(
        test,
        members=members, transactions=transactions, user_logs=user_logs)

    # Add members info
    train_useful = add_members_info(train_useful, members)
    test_useful = add_members_info(test_useful, members)

    # Add transactions info
    train_useful = add_transactions_info(train_useful, transactions)
    test_useful = add_transactions_info(test_useful, transactions)

    # Add user_logs info
    train_useful = add_user_logs_info(train_useful, user_logs)
    test_useful = add_user_logs_info(test_useful, user_logs)
