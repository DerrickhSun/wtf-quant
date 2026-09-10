import numpy as np
import pandas as pd
from scipy import stats

def historical_var(returns: pd.Series,
        confidence: float = 0.95,
        horizon_days: int = 1) -> float:
    """
    Historical simulation VaR.
    No distribution assumption — uses the empirical return distribution.
    """
    scaled = returns * np.sqrt(horizon_days)
    return float(np.percentile(scaled, (1 - confidence) * 100))

def parametric_var(returns: pd.Series,
        confidence: float = 0.95,
        horizon_days: int = 1) -> float:
    """
    Parametric VaR assuming normal distribution.
    mu and sigma estimated from historical returns.
    """
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - confidence)
    return float((mu + z * sigma) * np.sqrt(horizon_days))

def cvar(returns: pd.Series,
         confidence: float = 0.95,
         horizon_days: int = 1) -> float:
    """
    Conditional VaR (Expected Shortfall).
    Average of all returns below the VaR threshold.
    """
    var = historical_var(returns, confidence, horizon_days)
    tail = returns[returns <= var / np.sqrt(horizon_days)]
    return float(tail.mean() * np.sqrt(horizon_days))

def compute_risk_metrics(prices: pd.Series,
                         risk_free_rate: float = 0.05) -> dict:
    returns = prices.pct_change().dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Annualised metrics
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_vol

    # Drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    return {
        'annualised_return': round(ann_return, 4),
        'annualised_volatility': round(ann_vol, 4),
        'sharpe_ratio': round(sharpe, 4),
        'max_drawdown': round(max_drawdown, 4),
        'var_95_1d': round(historical_var(returns, 0.95, 1), 4),
        'var_99_1d': round(historical_var(returns, 0.99, 1), 4),
        'cvar_95_1d': round(cvar(returns, 0.95, 1), 4),
        'skewness': round(float(returns.skew()), 4),
        'kurtosis': round(float(returns.kurtosis()), 4),
    }
