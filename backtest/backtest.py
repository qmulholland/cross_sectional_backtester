import pandas as pd


def backtest_decile(
    df: pd.DataFrame,
    signal_col: str,
    ret_col: str
) -> pd.Series:
    """
    Long top decile, short bottom decile, equal-weighted.
    """
    df = df.dropna(subset=[signal_col, ret_col]).copy()

    df["decile"] = (
        df.groupby("date")[signal_col]
        .transform(lambda x: pd.qcut(x, 10, labels=False))
    )

    long = df[df["decile"] == 9]
    short = df[df["decile"] == 0]

    daily_pnl = (
        long.groupby("date")[ret_col].mean()
        - short.groupby("date")[ret_col].mean()
    )

    return daily_pnl
