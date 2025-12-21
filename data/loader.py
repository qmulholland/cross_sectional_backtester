"""
Loader.py automates the retrieval and cleaning of
historical market data. It standardizes diverse ticker
data into a long-format DataFrame, ensuring that the 
downstream technical functions recieve consistent 
and predictable inputs.

In the future, can be expanded to support different/multiple
data sources beyond Yahoo Finance. It would increae the system's
speed and reliability, while also allowing the strategy to 
incorporate different types of assets.
"""

import pandas as pd         #imports pandas

import yfinance as yf       #imports Yahoo Finance API 
                            #to fetch market Data


def load_prices(            #Defines function to download price data
                            #for a list of stocks over specific timeframe

    tickers: list[str],             #List of stock symbols to download
    start_date: str,                #Beginning date for historical data
    end_date: str | None = None     #End date(optional, defaults to today if None)
) -> pd.DataFrame:                  #Returns cleaned pandas DataFrame


    data = yf.download(         #Fetches raw market data from Yahoo Finance
        tickers,                #for all requested tickers at once
        start=start_date,
        end=end_date,
        auto_adjust=False,      #Keeps raw and adjusted prices separate for accuracy
        group_by="ticker"       #Organizes the initial download by stock symbol
    )

    frames = []     #Initializes empty list to store individual stock dataframes     

    for ticker in tickers:          #Iterates through each ticker to process specific data  
        df = data[ticker].copy()    #Creates copy of the data for the stock
        df["ticker"] = ticker       #Adds a new column to identify which stock data belongs to
        frames.append(df)           #Adds individual stock's dataframe to the list

    prices = pd.concat(frames).reset_index()        #Merges all individual stock's dataframes
                                                    #into one single "long-format" table

    prices.columns = [      #Renames columns to standardized, lowercase format for easier coding

        "date", "open", "high", "low", "close", "adj_close", "volume", "ticker"
    ]

    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)      #Sorts the final table
                                                                                #by stock symbol then date
                                                                                #for chronological order

    return prices       #Returns the finished dataset 
