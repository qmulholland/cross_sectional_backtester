"""
Configuration module for cross_sectional_equity_signal project.

Centralizes all hyperparameters, paths, tickers, and other project-wide settings.
"""

# ----------------------
# Data Settings
# ----------------------
DATA_PATH = "data/raw"          # Folder to store raw downloaded data
PROCESSED_PATH = "data/processed"  # Folder for cleaned / feature-enriched data

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
BENCHMARK_TICKER = "SPY"        # Benchmark for excess return calculation

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

# ----------------------
# Feature Engineering Settings
# ----------------------
MOMENTUM_WINDOWS = [5, 10, 21]  # Rolling windows for momentum
VOLATILITY_WINDOWS = [5, 10, 21]  # Rolling windows for volatility

# ----------------------
# Target Settings
# ----------------------
FORWARD_RETURN_HORIZON = 5  # Number of days ahead for forward return

# ----------------------
# Portfolio / Backtest Settings
# ----------------------
TOP_PCT = 0.1              # Top / bottom percentile for long-short
TARGET_VOL = 0.02          # Daily volatility target for portfolio
TRANSACTION_COST_BPS = 5    # Basis points per trade
SLIPPAGE_BPS = 2           # Basis points per trade for slippage

# ----------------------
# Model Settings
# ----------------------
MODEL_TYPE = "linear"       # Options: "linear", "gbr" (gradient boosting)
LINEAR_WEIGHTS = None       # Optional weights for linear signal

# ----------------------
# Miscellaneous
# ----------------------
RANDOM_SEED = 42
