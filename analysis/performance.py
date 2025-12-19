"""
Performance Analysis for cross_sectional_equity_signal

Milestones implemented:
1. Mean daily return, Std Dev, Sharpe, Hit Rate
2. Transaction costs
3. Out-of-sample split (train/test by date)
4. Benchmark comparison (equal-weighted portfolio)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data.loader import load_prices
from features.technical import (
    compute_daily_returns,
    compute_momentum,
    compute_volatility,
    cross_sectional_zscore,
    generate_signal
)
from backtest.backtest import backtest_decile
from costs.costs import apply_transaction_costs


def compute_metrics(pnl: pd.Series):
    """Compute mean, std, Sharpe, hit rate."""
    mean = pnl.mean()
    std = pnl.std()
    sharpe = mean / std * np.sqrt(252)  # annualized Sharpe
    hit_rate = (pnl > 0).sum() / len(pnl)
    return mean, std, sharpe, hit_rate


def performance_analysis(tickers, start_date="2020-01-01", split_date="2023-01-01"):
    # Load data
    prices = load_prices(tickers, start_date=start_date)

    # Compute features and signals
    prices = compute_daily_returns(prices)
    prices = compute_momentum(prices)
    prices = compute_volatility(prices)

    features = ["mom_5", "mom_10", "mom_21", "vol_5", "vol_10", "vol_21"]
    prices = cross_sectional_zscore(prices, features)
    signal_features = [f"{f}_z" for f in features]
    prices = generate_signal(prices, signal_features)

    # Split in-sample and out-of-sample
    train = prices[prices["date"] < split_date].copy()
    test = prices[prices["date"] >= split_date].copy()

    print("=== IN-SAMPLE PERFORMANCE ===")
    pnl_train = backtest_decile(train, signal_col="pred_signal", ret_col="ret_1d")
    pnl_train = apply_transaction_costs(pnl_train, cost_bps=5)
    metrics_train = compute_metrics(pnl_train)
    print(f"Mean daily return: {metrics_train[0]:.5f}")
    print(f"Std dev: {metrics_train[1]:.5f}")
    print(f"Annualized Sharpe: {metrics_train[2]:.2f}")
    print(f"Hit rate: {metrics_train[3]*100:.1f}%")
    (pnl_train.cumsum()).plot(title="In-Sample Cumulative PnL", figsize=(10,5))
    plt.show()

    print("\n=== OUT-OF-SAMPLE PERFORMANCE ===")
    pnl_test = backtest_decile(test, signal_col="pred_signal", ret_col="ret_1d")
    pnl_test = apply_transaction_costs(pnl_test, cost_bps=5)
    metrics_test = compute_metrics(pnl_test)
    print(f"Mean daily return: {metrics_test[0]:.5f}")
    print(f"Std dev: {metrics_test[1]:.5f}")
    print(f"Annualized Sharpe: {metrics_test[2]:.2f}")
    print(f"Hit rate: {metrics_test[3]*100:.1f}%")
    (pnl_test.cumsum()).plot(title="Out-of-Sample Cumulative PnL", figsize=(10,5))
    plt.show()

    # Benchmark: equal-weight portfolio
    print("\n=== BENCHMARK: EQUAL-WEIGHT PORTFOLIO ===")
    ew = test.groupby("date")["ret_1d"].mean()
    metrics_ew = compute_metrics(ew)
    print(f"Mean daily return: {metrics_ew[0]:.5f}")
    print(f"Std dev: {metrics_ew[1]:.5f}")
    print(f"Annualized Sharpe: {metrics_ew[2]:.2f}")
    print(f"Hit rate: {metrics_ew[3]*100:.1f}%")
    (ew.cumsum()).plot(title="Benchmark Equal-Weight Cumulative PnL", figsize=(10,5))
    plt.show()


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOG"]  # Replace with full universe later
    performance_analysis(tickers)
