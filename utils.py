import numpy as np
import pandas as pd
import copy


def num_cols(df):
    types = [type(df[col].values[1]) for col in df.columns]
    is_num = [not int(x is str) for x in types]
    is_num_idx = np.flatnonzero(is_num)

    return is_num_idx


def any_str_col(df):
    types = [type(df[col].values[0]) for col in df.columns]
    is_str = [int(x is str) for x in types]
    is_str_tot = sum(is_str)

    any_str = is_str_tot > 0

    return any_str


def decode_bytes(ds):
    for col in ds.columns:
        col_str = ds[col].values

        if type(col_str[0]) == bytes:
            col_str = [x.decode('utf-8') for x in col_str]
            ds[col] = col_str
        else:
            pass

    return ds


def col_as_int(x):
    ux = np.unique(x)

    mydict = dict()
    for i, elem in enumerate(ux):
        mydict[elem] = i

    num_col = [mydict[elem] for elem in x]

    return num_col


def col_obj2num(x):
    cols = x.columns[x.dtypes.eq('object')]

    x[cols] = x[cols].apply(pd.to_numeric, errors='coerce')

    return x


def as_df(X, y):
    Xdf = pd.DataFrame(X, columns=["X" + str(i)
                                   for i in range(X.shape[1])])

    Xdf["target"] = y

    return Xdf


def comb_df(df1, df2):
    df1_ = copy.deepcopy(df1)
    df2_ = copy.deepcopy(df2)

    df3 = df1_.append(df2_, ignore_index=True)
    df3.reset_index(drop=True, inplace=True)
    return df3
