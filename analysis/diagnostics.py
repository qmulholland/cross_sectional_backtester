import pandas as pd
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


def run_diagnostics():
    tickers = ["AAPL", "MSFT", "GOOG"]

    prices = load_prices(tickers, start_date="2020-01-01")

    prices = compute_daily_returns(prices)
    prices = compute_momentum(prices)
    prices = compute_volatility(prices)

    features = ["mom_5", "mom_10", "mom_21", "vol_5", "vol_10", "vol_21"]
    prices = cross_sectional_zscore(prices, features)

    signal_features = [f"{f}_z" for f in features]
    prices = generate_signal(prices, signal_features)

    pnl = backtest_decile(
        prices,
        signal_col="pred_signal",
        ret_col="ret_1d"
    )

    print("Cumulative PnL:")
    print(pnl.cumsum().tail())

    pnl.cumsum().plot(title="Cross-Sectional Strategy PnL")
    plt.show()


if __name__ == "__main__":
    run_diagnostics()
