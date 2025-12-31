from engine.loader import load_prices
from strategy.momentum_vol import apply_strategy
from engine.backtest import run_backtest
from engine.output import generate_performance_report

# CONFIG
TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NVDA", "NFLX"] # Add your tickers
START_DATE = "2020-01-01"
SPLIT_DATE = "2023-01-01"
SPY_TICKER = "SPY"

def main():
    # Load and Process
    prices = load_prices(TICKERS, start_date=START_DATE)
    processed = apply_strategy(prices)

    # Split
    train = processed[processed["date"] < SPLIT_DATE].copy()
    test = processed[processed["date"] >= SPLIT_DATE].copy()

    # Backtest
    pnl_is = run_backtest(train)
    pnl_oos = run_backtest(test)

    # SPY Benchmark Data
    spy_prices = load_prices([SPY_TICKER], start_date=START_DATE)
    spy_prices["ret_1d"] = spy_prices.groupby("ticker")["adj_close"].pct_change()
    spy_returns = spy_prices[spy_prices["date"] >= SPLIT_DATE].set_index("date")["ret_1d"]

    # EXACT SAME OUTPUT
    generate_performance_report(pnl_is, pnl_oos, test, spy_returns)

if __name__ == "__main__":
    main()