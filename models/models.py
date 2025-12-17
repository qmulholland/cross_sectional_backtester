"""
Models Module

Contains predictive models for cross-sectional equity signals.
Currently supports:
- Linear weighted signal (baseline)
- Placeholder for ML models (XGBoost, LightGBM, scikit-learn)
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


def linear_weighted_signal(df: pd.DataFrame, feature_cols: list[str], weights: list[float] | None = None) -> pd.DataFrame:
    """
    Compute a linear weighted signal from z-scored features.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain feature columns
    feature_cols : list[str]
        Columns to use in the linear signal
    weights : list[float] or None
        Optional weights for each feature (equal if None)

    Returns
    -------
    pd.DataFrame
        Original df with 'pred_signal' column
    """
    df = df.copy()
    if weights is None:
        weights = [1.0] * len(feature_cols)
    if len(weights) != len(feature_cols):
        raise ValueError("Length of weights must match number of features")

    df['pred_signal'] = sum(df[f] * w for f, w in zip(feature_cols, weights))
    return df


def train_ml_model(df: pd.DataFrame, feature_cols: list[str], target_col: str, model_type='linear') -> tuple[pd.DataFrame, object]:
    """
    Train a predictive model (linear or tree-based) on historical features.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain feature_cols and target_col
    feature_cols : list[str]
        Predictor columns
    target_col : str
        Column to predict (e.g., forward returns)
    model_type : str
        'linear' or 'gbr' (gradient boosting regressor)

    Returns
    -------
    df : pd.DataFrame
        Original df with predicted values in 'pred_signal'
    model : trained sklearn model
    """
    df = df.copy()
    df = df.dropna(subset=feature_cols + [target_col])

    X = df[feature_cols].values
    y = df[target_col].values

    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'gbr':
        model = GradientBoostingRegressor(n_estimators=200, max_depth=3)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Time-series split for fitting
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Split Test MSE: {mean_squared_error(y_test, y_pred):.6f}")

    # Fit on full dataset
    model.fit(X, y)
    df['pred_signal'] = model.predict(X)

    return df, model
