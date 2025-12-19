import pandas as pd
import yfinance as yf


def load_prices(
    tickers: list[str],
    start_date: str,
    end_date: str | None = None
) -> pd.DataFrame:
    """
    Download adjusted OHLCV data and return a clean long-format DataFrame.
    """
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        group_by="ticker"
    )

    frames = []
    for ticker in tickers:
        df = data[ticker].copy()
        df["ticker"] = ticker
        frames.append(df)

    prices = pd.concat(frames).reset_index()

    prices.columns = [
        "date", "open", "high", "low", "close", "adj_close", "volume", "ticker"
    ]

    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    return prices
