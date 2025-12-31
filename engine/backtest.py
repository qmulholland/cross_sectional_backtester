import pandas as pd

def run_backtest(df: pd.DataFrame, signal_col: str = "pred_signal", cost_bps: int = 5) -> pd.Series:
    df = df.dropna(subset=[signal_col, "ret_1d"]).copy()
    df["decile"] = df.groupby("date")[signal_col].transform(lambda x: pd.qcut(x, 10, labels=False))
    
    long = df[df["decile"] == 9]
    short = df[df["decile"] == 0]
    
    daily_pnl = long.groupby("date")["ret_1d"].mean() - short.groupby("date")["ret_1d"].mean()
    
    # Apply exactly as costs.py did
    cost = cost_bps / 10000
    return daily_pnl - cost