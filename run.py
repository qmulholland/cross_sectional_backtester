"""
run.py acts as a convenient shortcut for the entire project.
"""

#Imports main analysis function from performance module
from analysis.performance import performance_analysis   

if __name__ == "__main__":  #Defines entry point for execution

    tickers = ["AAPL", "MSFT", "NVDA"]      #List of tickers, Update

    #Executes full pipeline
    performance_analysis(
        tickers=tickers,            #Tickers taken from list above
        start_date="2020-01-01",    #Date data starts at
        split_date="2023-01-01"     #"Splits" IS and OOS
    )