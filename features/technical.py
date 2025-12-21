"""
Technical.py serves as the mathematical foundation of the project by
transforming raw price data into statistical features. It calculates
momentum, volatility, and cross-sectional Z-scores. It creates a 
standardized field where stocks can be compared directly regardless
of their nominal price levels.

In the future, it can be expendad to include more complex technical
indicates (RSI, Mean Reversion Signals, or Volume-Weighted Metrics). 
This would create a multi-dimensional view of market behavior and
potentially increase the predictive accuracy of the final signal.
"""
import pandas as pd         #imports pandas
import numpy as np          #imports numpy


def compute_daily_returns(df: pd.DataFrame) -> pd.DataFrame:        #Calculates the 1-day percentage change 
                                                                    #for each stock's adjusted close price

    df = df.copy()              #Creates a copy of the DataFrame to 
                                #avoid modifying the input data directly

    df["ret_1d"] = df.groupby("ticker")["adj_close"].pct_change()       #Computes return per stock

    return df                                                           #Returns the updated DataFrame with the new return column


def compute_momentum(df: pd.DataFrame, windows=(5, 10, 21)) -> pd.DataFrame:   #Measures price momentum over specific 
                                                                                #lookback windows (e.g., 5, 10, 21 days)

    df = df.copy()          #Creates a copy of the DataFrame to avoid modifying the input data directly

    for w in windows:           #Interates through each defined time window

        df[f"mom_{w}"] = (
            df.groupby("ticker")["adj_close"]       #Groups by stock ticker
            .pct_change(w)                          #Calculates the percentage change over 'w' days
        )
    return df           #Returns the DataFrame with multiple momentum features


def compute_volatility(df: pd.DataFrame, windows=(5, 10, 21)) -> pd.DataFrame:      #Calculates the rolling standard deviation 
                                                                                    #of returns to measure risk and volatility
    df = df.copy()          #Creates a copy of the DataFrame to avoid modifying the input data directly

    for w in windows:       #Iterates through each defined time window

        df[f"vol_{w}"] = (
            df.groupby("ticker")["ret_1d"]          #Groups by ticker to isolate returns
            .rolling(w)                             #Creates a sliding window of 'w' days
            .std()                                  #Computes the standard deviation of returns in that window
            .reset_index(level=0, drop=True)        #Cleans the index for merging
        )
    return df           #Returns the DataFrame with volatility indicators


def cross_sectional_zscore(         #Standardizes features across all stocks for a given
    df: pd.DataFrame,               #date using Z-score normalization
    cols: list[str]
) -> pd.DataFrame:

    df = df.copy()          #Creates a copy of the DataFrame to avoid modifying the input data directly

    for col in cols:                #Iterates through the features to be normalized
        df[f"{col}_z"] = (
            df.groupby("date")[col]                             #Groups by date to compare stocks on that day
            .transform(lambda x: (x - x.mean()) / x.std())      #Calculates (Value - Mean) / Std
        )
    return df       #Returns data where features have a mean of 0 and std of 1


def generate_signal(                #Combines various Z-scored features into
    df: pd.DataFrame,               #a single composite predictive signal
    feature_cols: list[str]
) -> pd.DataFrame:

    df = df.copy()      #Creates a copy of the DataFrame to avoid modifying the input data directly

    df["pred_signal"] = df[feature_cols].mean(axis=1)       #Averages the chosen features row-wise

    return df   #Returns the final signal used by the backtester

