"""
Performance.py serves as "control center" for the 
entire project. It takes in raw market data and 
produces graphs for In-sample and Out-of-Sample
performances, Equal-Weight and SPY Benchmarks, 
and then a final graph for Comparison overlay
between Out-Of-Sample, Equal-Weight, and the SPY.
"""

import pandas as pd     #imports pandas (data structures/time-series manipulation)
import numpy as np      #imports numpy (mathmatical operations and square root functions)
import matplotlib.pyplot as plt     #imports plotting tools for visual performance analysis

from data.loader import load_prices     #Imports data fetching module
from features.technical import (        #Imports all technical indicator and signal functions
    compute_daily_returns,
    compute_momentum,
    compute_volatility,
    cross_sectional_zscore,
    generate_signal
)
from backtest.backtest import backtest_decile       #Imports ranking and execution logic
from costs.costs import apply_transaction_costs     #Imports fee deduction logic

SPY_TICKER = "SPY"  # Add SPY ticker to use as market benchmark

def compute_metrics(pnl: pd.Series):    
    #Function to calculate statistical risk and return metrics for series of returns  
   
    mean = pnl.mean()       #Average Daily Return
    std = pnl.std()         #Standard Deviation of Daily Returns (Volatility)
    sharpe = mean / std * np.sqrt(252)      #Annualized risk-adjusted return ratio
    hit_rate = (pnl > 0).sum() / len(pnl)   #Percentage of days with positive returns
    return mean, std, sharpe, hit_rate      #Returns metrics

def print_metrics_summary(label, pnl_series, initial_capital=1000):     
    #Formats and prints detailed performance report including capital growth and risk

    mean = pnl_series.mean()        #Average Daily Return   
    std = pnl_series.std()          #Standard Deviation of Daily Returns (Volatility)
    sharpe = (mean / std) * np.sqrt(252)        #Annualized risk-adjusted return ratio
    hit_rate = (pnl_series > 0).mean() * 100       #Percentage of days positive
    
    # Calculates the sequence of portfolio values over time assuming reinvestment
    cum_returns = (1 + pnl_series).cumprod()
    final_value = initial_capital * cum_returns.iloc[-1]
    
    # Measures the 'pain' of the strategy by finding largest peak-to-trough drop
    running_max = cum_returns.cummax()          #Tracks highest value reached 
    drawdown = (cum_returns - running_max) / running_max        #Daily percentage drop from peak
    max_drawdown = drawdown.min() * 100  # Captures worst-case loss scenario
    
    print(f"=== {label} ===")
    print(f"Annualized Sharpe: {sharpe:.2f}")
    print(f"Hit rate: {hit_rate:.1f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Compounded Growth (Start $1000): ${final_value:,.2f}")
    print()


def performance_analysis(tickers, start_date="2020-01-01", split_date="2023-01-01"):
    #  Orchestrates entire pipeline: data, features, backtesting, visualization

    #Data Acquisition and Features
    prices = load_prices(tickers, start_date=start_date)
    prices = compute_daily_returns(prices)
    prices = compute_momentum(prices)
    prices = compute_volatility(prices)
    
    #Signal Generation for cross-sectional Z-scoring
    features = ["mom_5", "mom_10", "mom_21", "vol_5", "vol_10", "vol_21"]
    prices = cross_sectional_zscore(prices, features)
    signal_features = [f"{f}_z" for f in features]
    prices = generate_signal(prices, signal_features)

    #Splits data into In-Sample (training) and Out-of-Sample (testing) period
    train = prices[prices["date"] < split_date].copy()
    test = prices[prices["date"] >= split_date].copy()

    #Backtests the training period with Transaction Costs Applied
    pnl_train = backtest_decile(train, signal_col="pred_signal", ret_col="ret_1d")
    pnl_train = apply_transaction_costs(pnl_train, cost_bps=5)
    print_metrics_summary("IN-SAMPLE PERFORMANCE", pnl_train)
    ((1 + pnl_train).cumprod() - 1).plot(title="In-Sample Compounded PnL", figsize=(10,5),
                                            ylabel="Compounded Total Return", xlabel="Date")
    plt.show()

    #Backtests the unseen test period to check for Strategy Robustness
    pnl_test = backtest_decile(test, signal_col="pred_signal", ret_col="ret_1d")
    pnl_test = apply_transaction_costs(pnl_test, cost_bps=5)
    print_metrics_summary("OUT-OF-SAMPLE PERFORMANCE", pnl_test)
    ((1 + pnl_test).cumprod() - 1).plot(title="Out-of-Sample Compounded PnL", figsize=(10,5),
                                        ylabel="Compounded Total Return", xlabel="Date")
    plt.show()

    #Calculates simple Equal-Weight Benchmark for the Same Period
    ew = test.groupby("date")["ret_1d"].mean()
    print_metrics_summary("BENCHMARK: EQUAL-WEIGHT PORTFOLIO", ew)
    ((1 + ew).cumprod() - 1).plot(title="Equal-Weight Compounded PnL", figsize=(10,5),
                                    ylabel="Compounded Total Return", xlabel="Date")
    plt.show()

    #Downloads and Prepares SPY data as market benchmark
    spy_prices = load_prices([SPY_TICKER], start_date=start_date)
    spy_prices = compute_daily_returns(spy_prices)
    spy_test = spy_prices[spy_prices["date"] >= split_date].copy()
    spy_returns = spy_test.set_index("date")["ret_1d"]
    
    print_metrics_summary("BENCHMARK: SPY", spy_returns)
    ((1 + spy_returns).cumprod() - 1).plot(title="SPY Compounded Returns", figsize=(10,5),
                                            ylabel="Compounded Total Return", xlabel="Date")
    plt.show()

    #Generates Comparative Chart of All Three Return Streams
    cum_df = pd.DataFrame({
        "Strategy OOS": (1 + pnl_test).cumprod() - 1,
        "Equal-Weight": (1 + ew).cumprod() - 1,
        "SPY": (1 + spy_returns).cumprod() - 1,
    })
    cum_df.plot(title="Compounded Returns Comparison", figsize=(12,6), ylabel="Total Return")
    plt.axhline(0, color='black', lw=1, ls='--')
    plt.show()


#Execution Entry Point, Kept for Isolated Testing
if __name__ == "__main__":
    tickers = []      #Empty Tickers, Update if Testing Isolated
    performance_analysis(tickers)
