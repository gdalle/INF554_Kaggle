"""Creating features."""

import pandas as pd
# import numpy as np
import extraction as ex


def count_days(date_series, base_date=pd.Timestamp(2000, 1, 1)):
    """Count the days elapsed since the base date for a series of dates."""
    return (date_series - base_date).astype("timedelta64[D]")


def categorize(df, col):
    """Turn categorical variables into 0/1 columns for each possible value."""
    cat = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, cat], axis=1)
    df = df.drop(col, axis=1)
    return df


def get_useful_users(users_table, members, transactions, user_logs):
    """Get subset from train or test whose users appear in other tables."""
    users_table_useful = users_table[
        # users_table.index.isin(transactions["msno"]) &
        # users_table.index.isin(user_logs["msno"].astype(object)) &
        users_table.index.isin(members.index)
    ].copy()
    return users_table_useful


def add_members_info(users_table, members):
    """
    Extract info from members and add it to train or test.

    We assume the users in users_table are all "useful",
    ie present in the members database.
    """
    useful_members = members.loc[users_table.index, :]

    # Info from members table

    # City (categories)
    users_table["city"] = useful_members["city"]
    users_table = categorize(users_table, "city")
    # Gender (categories)
    users_table["gender"] = useful_members["gender"]
    users_table = categorize(users_table, "gender")
    # Registered via (categories)
    users_table["registered_via"] = useful_members["registered_via"]
    users_table = categorize(users_table, "registered_via")
    # Registration (measured in days since 01/01/2000)
    users_table["registration_init"] = \
        count_days(useful_members["registration_init_time"])
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
    train_useful = get_useful_users(train, members, transactions, user_logs)
    test_useful = get_useful_users(test, members, transactions, user_logs)

    # Add members info
    train_useful = add_members_info(train_useful, members)
    test_useful = add_members_info(test_useful, members)
