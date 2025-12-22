**Quantitative Cross-Sectional Backtester**
An end-to-end Python framework designed to transform
raw market data into a validated long-short trading 
strategy. This project uses a decile-ranking engine
that evaluates assets based on momentum and volatility
factors while accounting for transaction costs.

**Project Structure**
cross_sectional_backtester/
    analysis/
        - diagnostics.py    # 'Sanity check' to verify pipeline works
        - performance.py:   # Calculates the Sharpe Ratio, Max Drawdown, and visualizes PnL

    backtest/
        - backtest.py       # Executes the cross-sectional ranking and portfolio construction

    costs/
        - costs.py          # Models realistic performance by applying transaction fees

    data/
        - loader.py         # Ingests data from the Yahoo Finance API

    features/   
        - features.py       # Centralized export hub. Provides clean interface for rest of app to access indicators
        -technical.py:      # Mathematical engine where momentum, volatility, and Z-scores are generated

    portfolio/
        - portfolio.py      # Possible future "update". Research module for signal-proportional weighting

    run.py                  # The primary controller. Update ticker list and date to run analysis here.

**Installation and Setup**
- 1. Clone repository to local machine
- 2. Install dependencies using the requirements file:
        "pip install -r requirements.txt" into the terminal

**Usage**
    Running the Full Analysis:
        "python run.py" into the terminal
    Running a Diagnostic Test:
        "python -m analysis.diagnostics" into the terminal

**Strategy Methodology**
1. Signal Processing: Raw price data is converted into technical features (5, 10, and 21-day windows)
2. Standardization: Feature are Z-scored cross-sectionally to compare stocks of different price levels fairly
3. Execution Logic: Each trading day, the system ranks the selected tickers and goes Long for the top 10% and Short for the bottom 10%
4. Validation: The strategy is split into In-Sample (Training) and Out-of-Sample (Testing) periods to check for overfitting