import pandas as pd


def equal_weight_portfolio(df: pd.DataFrame, signal_col: str):
    weights = (
        df.groupby("date")[signal_col]
        .transform(lambda x: x / x.abs().sum())
    )
    df = df.copy()
    df["weight"] = weights
    return df
