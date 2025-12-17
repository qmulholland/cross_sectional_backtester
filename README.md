
# Cross-Sectional Equity Signal Project

## Overview


This project develops a **data-driven, cross-sectional equity strategy** to predict short-term 
stock returns.


It demonstrates skills in:

- Data acquisition and cleaning 
- Feature engineering (technical factors, z-scores)
- Target computation (forward and excess returns)
- Predictive modeling (linear, tree-based models)
- Portfolio construction and backtesting
- Transaction cost and slippage modeling

The pipeline is modular and designed for extensibility, allowing new features, models, and
backtesting rules to be easily added.

---

## Project Structure
cross_sectional_equity_signal/
│
├── data/   # Raw and processed data
│ ├── loader.py     # Data loading and cleaning
│
├── features/ # Feature engineering
│ ├── init.py
│ ├── features.py # Feature aggregator
│ └── technical.py # Technical feature functions
│
├── models/ # Predictive models
│ ├── init.py
│ └── models.py
│
├── portfolio.py  # Portfolio construction and PnL calculation
├── costs.py  # Transaction cost and slippage models
├── target.py  # Forward and excess return computation
├── universe.py  # Universe selection and filtering
├── config.py  # Project configuration / hyperparameters
├── diagnostics.py  # Pipeline and data diagnostics
├── requirements.txt  # Python dependencies
└── README.md  # This file


---

**Installation**

1. Clone the repository

git clone https://github.com/<your-username>/cross_sectional_equity_signal.git
cd cross_sectional_equity_signal

2. Create and Activate a Virtual Environment
python3 -m venv venv
source venv/bin/activate
venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

**Usage**
1. Load Data
from data.loader import load_prices
prices = load_prices(tickers=["AAPL", "MSFT", "GOOG"], start_date="2020-01-01")

2. Compute Features
from features.features import compute_all_features
prices = compute_all_features(prices)

3. Compute Target
from target import compute_forward_returns
prices = compute_forward_returns(prices, horizon=5)

4. Generate Portfolio
from portfolio import equal_weight_portfolio, compute_portfolio_pnl
portfolio_df = equal_weight_portfolio(prices, signal_col="pred_signal")
daily_pnl = compute_portfolio_pnl(portfolio_df)

5. Apply Costs and Slippage
from costs import apply_transaction_costs, apply_slippage
portfolio_df = apply_transaction_costs(portfolio_df, cost_per_trade=0.0005)
portfolio_df = apply_slippage(portfolio_df, slippage_bps=0.0002)

6. Backtesting
import matplotlib.pyplot as plt
daily_pnl.cumsum().plot(title="Cumulative PnL")
plt.show()

**License**
This project is for educational purposes and research. Modify and use at your own discretion. 