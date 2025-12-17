"""
Portfolio Module

Handles position sizing, portfolio construction, and basic risk management
for cross-sectional equity strategies.
"""

import pandas as pd
import numpy as np


def equal_weight_portfolio(df: pd.DataFrame, signal_col='pred_signal', top_pct=0.1) -> pd.DataFrame:
    """
    Construct an equal-weight long-short portfolio based on predicted signals.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'date', 'ticker', and signal_col
    signal_col : str
        Column name for the signal to rank tickers
    top_pct : float
        Fraction of tickers to long/short (e.g., 0.1 for deciles)

    Returns
    -------
    pd.DataFrame
        Same df with new column 'weight' for portfolio positions
    """
    df = df.copy()

    # Rank tickers by signal within each date
    df['rank'] = df.groupby('date')[signal_col].rank(method='first', ascending=False)
    df['n_tickers'] = df.groupby('date')['ticker'].transform('count')

    # Initialize weights
    df['weight'] = 0.0
    df.loc[df['rank'] <= df['n_tickers'] * top_pct, 'weight'] = 1.0  # Long top decile
    df.loc[df['rank'] > df['n_tickers'] * (1 - top_pct), 'weight'] = -1.0  # Short bottom decile

    # Normalize weights by total absolute value
    df['weight'] = df.groupby('date')['weight'].transform(lambda x: x / x.abs().sum())

    return df


def volatility_targeted_portfolio(df: pd.DataFrame, signal_col='pred_signal', top_pct=0.1, target_vol=0.02, ret_col='ret_1d') -> pd.DataFrame:
    """
    Construct a portfolio with position sizing scaled by recent volatility.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'date', 'ticker', signal_col, and ret_col
    signal_col : str
        Column name for predicted signal
    top_pct : float
        Fraction of tickers to long/short
    target_vol : float
        Target daily portfolio volatility (e.g., 2%)
    ret_col : str
        Column name for daily returns

    Returns
    -------
    pd.DataFrame
        Same df with 'weight' scaled to target volatility
    """
    df = equal_weight_portfolio(df, signal_col=signal_col, top_pct=top_pct)
    df = df.copy()

    # Estimate daily volatility per ticker (rolling 21-day std)
    df['vol'] = df.groupby('ticker')[ret_col].transform(lambda x: x.rolling(21).std())

    # Scale weights to target volatility
    df['weight'] = df['weight'] * target_vol / df['vol']

    # Cap extreme weights
    df['weight'] = df['weight'].clip(-0.1, 0.1)  # optional, max 10% per ticker

    return df


def compute_portfolio_pnl(df: pd.DataFrame, ret_col='ret_1d') -> pd.Series:
    """
    Compute daily portfolio P&L given positions and returns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'date', 'weight', and ret_col
    ret_col : str
        Column for daily returns

    Returns
    -------
    pd.Series
        Daily P&L of the portfolio
    """
    df = df.copy()
    df['pnl'] = df['weight'] * df[ret_col]
    daily_pnl = df.groupby('date')['pnl'].sum()
    return daily_pnl
