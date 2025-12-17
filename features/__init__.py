"""
Features Package

Provides access to all feature engineering modules.
"""

from . import features
from . import technical

# Optional: expose common functions at the package level
from .technical import (
    compute_daily_returns,
    compute_momentum,
    compute_volatility,
    cross_sectional_zscore,
    generate_signal
)
from .features import compute_all_features
