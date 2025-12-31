import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def print_metrics_summary(label, pnl_series, initial_capital=1000):
    mean = pnl_series.mean()
    std = pnl_series.std()
    sharpe = (mean / std) * np.sqrt(252)
    hit_rate = (pnl_series > 0).mean() * 100
    
    cum_returns = (1 + pnl_series).cumprod()
    final_value = initial_capital * cum_returns.iloc[-1]
    
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    print(f"=== {label} ===")
    print(f"Annualized Sharpe: {sharpe:.2f}")
    print(f"Hit rate: {hit_rate:.1f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Compounded Growth (Start $1000): ${final_value:,.2f}")
    print()

def generate_performance_report(pnl_is, pnl_oos, test_data, spy_returns):
    # 1. In-Sample Plot
    ((1 + pnl_is).cumprod() - 1).plot(title="In-Sample Compounded PnL", figsize=(10,5), ylabel="Compounded Total Return", xlabel="Date")
    print_metrics_summary("IN-SAMPLE PERFORMANCE", pnl_is)
    plt.show()

    # 2. Out-of-Sample Plot
    ((1 + pnl_oos).cumprod() - 1).plot(title="Out-of-Sample Compounded PnL", figsize=(10,5), ylabel="Compounded Total Return", xlabel="Date")
    print_metrics_summary("OUT-OF-SAMPLE PERFORMANCE", pnl_oos)
    plt.show()

    # 3. Equal-Weight Benchmark
    ew = test_data.groupby("date")["ret_1d"].mean()
    ((1 + ew).cumprod() - 1).plot(title="Equal-Weight Compounded PnL", figsize=(10,5), ylabel="Compounded Total Return", xlabel="Date")
    print_metrics_summary("BENCHMARK: EQUAL-WEIGHT PORTFOLIO", ew)
    plt.show()

    # 4. SPY Benchmark
    ((1 + spy_returns).cumprod() - 1).plot(title="SPY Compounded Returns", figsize=(10,5), ylabel="Compounded Total Return", xlabel="Date")
    print_metrics_summary("BENCHMARK: SPY", spy_returns)
    plt.show()

    # 5. Final Comparison Chart
    cum_df = pd.DataFrame({
        "Strategy OOS": (1 + pnl_oos).cumprod() - 1,
        "Equal-Weight": (1 + ew).cumprod() - 1,
        "SPY": (1 + spy_returns).cumprod() - 1,
    })
    cum_df.plot(title="Compounded Returns Comparison", figsize=(12,6), ylabel="Total Return")
    plt.axhline(0, color='black', lw=1, ls='--')
    plt.show()