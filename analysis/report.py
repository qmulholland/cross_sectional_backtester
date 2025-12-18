"""
Automated Performance Report for cross_sectional_equity_signal
Generates CSV metrics and PNG plots for:
- IS/OOS performance
- Benchmarks (Equal-weight, SPY)
- Turnover and transaction costs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pathlib import Path

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

# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(pnl: pd.Series):
    mean = pnl.mean()
    std = pnl.std()
    sharpe = mean / std * np.sqrt(252)
    hit_rate = (pnl > 0).sum() / len(pnl)
    return mean, std, sharpe, hit_rate

# -----------------------------
# Generate report
# -----------------------------
def generate_report(tickers, start_date="2020-01-01", split_date="2023-01-01", out_dir="report"):
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)

    # Load and preprocess prices
    prices = load_prices(tickers, start_date=start_date)
    prices = compute_daily_returns(prices)
    prices = compute_momentum(prices)
    prices = compute_volatility(prices)

    features = ["mom_5", "mom_10", "mom_21", "vol_5", "vol_10", "vol_21"]
    prices = cross_sectional_zscore(prices, features)
    signal_features = [f"{f}_z" for f in features]
    prices = generate_signal(prices, signal_features)

    # Split IS/OOS
    train = prices[prices['date'] < split_date].copy()
    test = prices[prices['date'] >= split_date].copy()

    metrics_dict = {}

    # -----------------------------
    # In-sample
    pnl_train = backtest_decile(train, signal_col="pred_signal", ret_col="ret_1d")
    pnl_train = apply_transaction_costs(pnl_train, cost_bps=5)
    metrics_dict['IS'] = compute_metrics(pnl_train)
    (pnl_train.cumsum()).plot(title="In-Sample Cumulative PnL")
    plt.savefig(out_path / "IS_cum_pnl.png")
    plt.close()

    # -----------------------------
    # Out-of-sample
    pnl_test = backtest_decile(test, signal_col="pred_signal", ret_col="ret_1d")
    pnl_test = apply_transaction_costs(pnl_test, cost_bps=5)
    metrics_dict['OOS'] = compute_metrics(pnl_test)
    (pnl_test.cumsum()).plot(title="Out-of-Sample Cumulative PnL")
    plt.savefig(out_path / "OOS_cum_pnl.png")
    plt.close()

    # -----------------------------
    # Turnover
    positions = pd.DataFrame(0, index=test['date'].unique(), columns=tickers)
    top_decile = test[test['pred_signal'] > test['pred_signal'].quantile(0.9)]
    bottom_decile = test[test['pred_signal'] < test['pred_signal'].quantile(0.1)]
    for t in tickers:
        positions.loc[top_decile.loc[top_decile['ticker']==t, 'date'], t] = 1
        positions.loc[bottom_decile.loc[bottom_decile['ticker']==t, 'date'], t] = -1
    turnover = positions.diff().abs().sum(axis=1).mean()

    # -----------------------------
    # Equal-weight benchmark
    ew = test.groupby("date")["ret_1d"].mean()
    metrics_dict['Equal-Weight'] = compute_metrics(ew)
    (ew.cumsum()).plot(title="Equal-Weight Benchmark")
    plt.savefig(out_path / "equal_weight.png")
    plt.close()

    # -----------------------------
    # SPY benchmark
    spy = yf.download("SPY", start=test['date'].min())
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [' '.join(col).strip() for col in spy.columns.values]
    if 'Adj Close' in spy.columns:
        spy_prices = spy['Adj Close']
    else:
        spy_prices = spy.iloc[:, 3]
    spy['ret_1d'] = spy_prices.pct_change()
    spy = spy.loc[test['date'].min(): test['date'].max()]
    metrics_dict['SPY'] = compute_metrics(spy['ret_1d'].dropna())
    (spy['ret_1d'].cumsum()).plot(title="SPY Benchmark")
    plt.savefig(out_path / "SPY.png")
    plt.close()

    # -----------------------------
    # Save metrics to CSV
    metrics_df = pd.DataFrame.from_dict(
        metrics_dict, orient='index', columns=['mean', 'std', 'sharpe', 'hit_rate']
    )

    # Add turnover as a separate row
    metrics_df.loc['Turnover'] = [np.nan, np.nan, np.nan, np.nan]
    metrics_df.loc['Turnover', 'turnover'] = turnover if 'turnover' in metrics_df.columns else turnover

    metrics_df.to_csv(out_path / "performance_metrics.csv")
    print(f"Report generated in {out_path}/")
    print(metrics_df)

# -----------------------------
# Run example
# -----------------------------
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOG"]  # Replace with full universe
    generate_report(tickers)
