import pandas as pd

def load_prices():
    """
    Stub function to simulate loading price data.
    Returns a small test DataFrame.
    """
    data = {
        "Date": ["2025-12-16", "2025-12-17", "2025-12-18"],
        "Ticker": ["AAPL", "MSFT", "GOOG"],
        "Close": [180.5, 345.2, 2900.3]
    }
    df = pd.DataFrame(data)
    print("Price data loaded:")
    print(df)
    return df

# Test the function if this file is run directly
if __name__ == "__main__":
    load_prices()
