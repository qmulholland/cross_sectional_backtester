import pandas as pd
import numpy as np


def compute_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_1d"] = df.groupby("ticker")["adj_close"].pct_change()
    return df


def compute_momentum(df: pd.DataFrame, windows=(5, 10, 21)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"mom_{w}"] = (
            df.groupby("ticker")["adj_close"]
            .pct_change(w)
        )
    return df


def compute_volatility(df: pd.DataFrame, windows=(5, 10, 21)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"vol_{w}"] = (
            df.groupby("ticker")["ret_1d"]
            .rolling(w)
            .std()
            .reset_index(level=0, drop=True)
        )
    return df


def cross_sectional_zscore(
    df: pd.DataFrame,
    cols: list[str]
) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[f"{col}_z"] = (
            df.groupby("date")[col]
            .transform(lambda x: (x - x.mean()) / x.std())
        )
    return df


def generate_signal(
    df: pd.DataFrame,
    feature_cols: list[str]
) -> pd.DataFrame:
    df = df.copy()
    df["pred_signal"] = df[feature_cols].mean(axis=1)
    return df
