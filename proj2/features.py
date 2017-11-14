"""Creating features."""

import pandas as pd
import numpy as np
import extraction as ex


def count_days(date_series, base_date=pd.Timestamp(2000, 1, 1)):
    """Count the days elapsed since the base date for a series of dates."""
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
            users_table.index.isin(transactions["msno"])
    if user_logs is not None:
        print("- User logs")
        useful_indices = useful_indices & \
            users_table.index.isin(user_logs["msno"].astype(str))
    return users_table[useful_indices]


def exploit_members(members):
    """Extract relevant info from members and put it in a DataFrame."""
    msno = members.index.unique()

    # Exploring members
    modified_members = members.drop("registration_init_time", axis=1)

    # City (categories)
    modified_members = categorize(modified_members, "city")
    # Bd (categories)
    modified_members = categorize(modified_members, "bd")
    # Gender (categories)
    modified_members = categorize(modified_members, "gender")
    # Registered via (categories)
    modified_members = categorize(modified_members, "registered_via")

    # Registration init time
    registration_init = pd.DataFrame(
        data=count_days(members["registration_init_time"]),
        index=modified_members.index,
        columns=["registration_init_time"]
    )

    # Reindex and concatenate
    members_data = [modified_members, registration_init]
    members_data = [df.reindex(msno) for df in members_data]

    return pd.concat(members_data, axis=1)


def exploit_transactions(transactions):
    """Extract relevant info from transactions and put it in a DataFrame."""
    msno = transactions["msno"].unique()

    # Exploring transactions
    grouped_trans = transactions.groupby("msno")

    # Latest transactions and planned expiration
    latest_trans = grouped_trans.max()
    latest_trans = latest_trans.loc[:, ["transaction_date", "membership_expire_date"]]
    latest_trans = latest_trans.apply(count_days)
    latest_trans.columns = ["latest_transaction_date", "planned_membership_expire_date"]

    # Transaction duration
    trans_dates = transactions.loc[:, ["msno", "membership_expire_date", "transaction_date"]]
    trans_dur = trans_dates["membership_expire_date"]-trans_dates["transaction_date"]
    trans_dur = count_days(trans_dur)
    trans_dates = trans_dates.assign(mean_transaction_duration=trans_dur)
    mean_trans_dates = trans_dates.groupby("msno").mean()
    mean_trans_dur = mean_trans_dates.loc[:, ["mean_transaction_duration"]]

    # Auto-renew and cancel
    trans_caracs = grouped_trans.mean()
    trans_caracs = trans_caracs.loc[:, ["is_auto_renew", "is_cancel"]]
    trans_caracs.columns = ["auto_renew_freq", "cancel_freq"]

    # Reindex and concatenate
    transactions_data = [latest_trans, mean_trans_dur, trans_caracs]
    transactions_data = [df.reindex(msno) for df in transactions_data]

    return pd.concat(transactions_data, axis=1)


def exploit_user_logs(user_logs):
    """Extract relevant info from user_logs and put it in a DataFrame."""
    msno = user_logs["msno"].unique()

    # Exploring user logs
    grouped_logs = user_logs.groupby("msno")

    if "num_25" in user_logs.columns:
        # Mean logs
        mean_logs = grouped_logs.mean()
        cols = mean_logs.columns
        mean_logs.columns = ["mean_" + col for col in cols]

    # Latest listening session
    latest_session = grouped_logs.max().loc[:, ["date"]]
    latest_session = latest_session.apply(count_days)
    latest_session.columns = ["latest_listening_session"]

    # Number of listening sessions
    logs_count = grouped_logs.count().loc[:, ["date"]]
    logs_count.columns = ["number_listening_sessions"]

    # Reindex and concatenate
    user_logs_data = [latest_session, logs_count]
    if "num_25" in user_logs.columns:
        user_logs_data += [mean_logs]
    user_logs_data = [df.reindex(msno) for df in user_logs_data]

    return pd.concat(user_logs_data, axis=1)


def add_data_to_users(users_table, df_list):
    """Concatenate a list of DF and add the information for each user."""
    users_table_full = users_table.copy()
    for df in df_list:
        df = df.reindex(users_table.index)
        users_table_full = pd.concat([users_table_full, df], axis=1)
    return users_table_full


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
