import pandas as pd
import numpy as np

def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_1d"] = df.groupby("ticker")["adj_close"].pct_change()
    
    windows = (5, 10, 21)
    for w in windows:
        df[f"mom_{w}"] = df.groupby("ticker")["adj_close"].pct_change(w)
        df[f"vol_{w}"] = df.groupby("ticker")["ret_1d"].rolling(w).std().reset_index(level=0, drop=True)

    features = ["mom_5", "mom_10", "mom_21", "vol_5", "vol_10", "vol_21"]
    for col in features:
        df[f"{col}_z"] = df.groupby("date")[col].transform(lambda x: (x - x.mean()) / x.std())

    signal_features = [f"{f}_z" for f in features]
    df["pred_signal"] = df[signal_features].mean(axis=1)
    return df