
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

class DataframeFunctionTransformer():
    def __init__(self, func):
        self.func = func

    def transform(self, input_df, **transform_params):
        return self.func(input_df)

    def fit(self, X, y=None, **fit_params):
        return self

def step_day_extractor(input_df):
    input_df["stepDay"] = input_df["step"].map(lambda t: np.ceil(t/24))
    return input_df

def step_hour_extractor(input_df):
    input_df["stepHour"] = input_df["step"].map(lambda t: t - (np.ceil(t/24) - 1) * 24)
    return input_df

def step_weekday_extractor(input_df):
    input_df["stepWeekDay"] = input_df["stepDay"].map(lambda t: np.remainder(t, 7))
    return input_df

def is_dest_merchant_extractor(input_df):
    input_df["isDestMerchant"] = input_df["nameDest"].map(lambda t: (t[0] == "M") )
    input_df["isDestMerchant"] = input_df["isDestMerchant"].astype("int")
    return input_df

def is_dest_balance_new_zero(input_df):
    input_df["isDestBalanceNewZero"] = input_df["newbalanceDest"].map(lambda t: t == 0 )
    input_df["isDestBalanceNewZero"] = input_df["isDestBalanceNewZero"].astype("int")
    return input_df

def is_dest_balance_old_zero(input_df):
    input_df["isDestBalanceOldZero"] = input_df["oldbalanceDest"].map(lambda t: t == 0 )
    input_df["isDestBalanceOldZero"] = input_df["isDestBalanceOldZero"].astype("int")
    return input_df

def is_orig_balance_new_zero(input_df):
    input_df["isOrigBalanceNewZero"] = input_df["newbalanceOrig"].map(lambda t: t == 0 )
    input_df["isOrigBalanceNewZero"] = input_df["isOrigBalanceNewZero"].astype("int")
    return input_df

def is_orig_balance_old_zero(input_df):
    input_df["isOrigBalanceOldZero"] = input_df["oldbalanceOrg"].map(lambda t: t == 0 )
    input_df["isOrigBalanceOldZero"] = input_df["isOrigBalanceOldZero"].astype("int")
    return input_df


FeatPipeline = Pipeline([
    ('step_day_extractor' ,DataframeFunctionTransformer(step_day_extractor)),
    ('step_hour_extractor' ,DataframeFunctionTransformer(step_hour_extractor)),
    ('step_weekday_extractor' ,DataframeFunctionTransformer(step_weekday_extractor)),
    ('is_dest_merchant_extractor' ,DataframeFunctionTransformer(is_dest_merchant_extractor)),
    ('is_dest_balance_new_zero' ,DataframeFunctionTransformer(is_dest_balance_new_zero)),
    ('is_dest_balance_old_zero' ,DataframeFunctionTransformer(is_dest_balance_old_zero)),
    ('is_orig_balance_new_zero' ,DataframeFunctionTransformer(is_orig_balance_new_zero)),
    ('is_orig_balance_old_zero' ,DataframeFunctionTransformer(is_orig_balance_old_zero))
])

