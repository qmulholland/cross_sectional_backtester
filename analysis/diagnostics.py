"""

"""

import pandas as pd                 #imports pandas
import matplotlib.pyplot as plt     #imports plotting tools for visual verification

from data.loader import load_prices     #Imports data fetching logic
from features.technical import (        #Imports technical processing and signal math
    compute_daily_returns,
    compute_momentum,
    compute_volatility,
    cross_sectional_zscore,
    generate_signal
)
from backtest.backtest import backtest_decile       #Imports core ranking engine


def run_diagnostics():
    tickers = ["AAPL", "MSFT", "GOOG"]      #Used as CONTROLLED TEST GROUP to ensure software is working

    prices = load_prices(tickers, start_date="2020-01-01")  #Pulls raw price data for diagnostic period

    #Exectures the full feature engineering chain to ensure no errors during math steps
    prices = compute_daily_returns(prices)     
    prices = compute_momentum(prices)
    prices = compute_volatility(prices)

    #Identifies features and normalizes them for cross-stock comparison
    features = ["mom_5", "mom_10", "mom_21", "vol_5", "vol_10", "vol_21"]
    prices = cross_sectional_zscore(prices, features)

    #Aggregates Z-scored features into single prediction signal
    signal_features = [f"{f}_z" for f in features]
    prices = generate_signal(prices, signal_features)

    #Runs simplified backtest on combined data without cost adjustments
    pnl = backtest_decile(
        prices,
        signal_col="pred_signal",
        ret_col="ret_1d"
    )

    #Prints last few values of the cumulative return to the console 
    print("Cumulative PnL:")
    print(pnl.cumsum().tail())

    #Generates basic plot to confirm signal is producing expected PnL behavior
    pnl.cumsum().plot(title="Cross-Sectional Strategy PnL")
    plt.show()


#Standard entry point to execute the diagnostic script independently
if __name__ == "__main__":
    run_diagnostics()
