from sklearn import preprocessing
import pandas as pd

data_home = "C:/Users/liadb/PycharmProjects/ass2-data"


def process_binary_cols(df):
    binary_cols = [col_name for col_name in df.columns if df[col_name].count() == 2]
    for col in binary_cols:
        values_set = set(df[col].tolist())
        new_value = 0
        for val in values_set:
            df[col] = df[col].replace([val], new_value)
            new_value += 1
    return df


def process_target_class(df, target_class):
    values_set = set(df[target_class].tolist())
    i = 0
    for val in values_set:
        df[target_class] = df[target_class].replace([val], i)
        i += 1
    return df


def encode_strings(df):
    le = preprocessing.LabelEncoder()
    for column_name in df.columns:
        if df[column_name].dtype == object:
            df[column_name] = le.fit_transform(df[column_name])
        else:
            pass
    return df


def handle_null(df):
    for col in df.columns:
        const = -1
        values_set = set(df[col].tolist())
        while const in values_set:
            const = const * 1000
        df[[col]].fillna(value=const, inplace=True)
    return df


def drop_cols(df, cols_to_drop):
    df = df.drop(columns=cols_to_drop)
    return df


def process_file(name, target_class, cols_to_drop=None):
    file_path = data_home + "/original" + "/" + name
    df = pd.read_csv(file_path)
    if cols_to_drop:
        df = drop_cols(df, cols_to_drop)
    df = process_target_class(df,target_class)
    df = process_binary_cols(df)
    df = encode_strings(df)
    df = handle_null(df)
    df.to_csv(data_home + "/processed" + "/" + name, index=False)
