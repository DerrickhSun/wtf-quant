import numpy as np
import pandas as pd
from scipy import stats

def rolling_volatility(returns: pd.Series,
                       windows: list[int] = [20, 60, 252]) -> pd.DataFrame:
    """
    Annualised rolling historical volatility for multiple windows.
    20-day = monthly, 60-day = quarterly, 252-day = annual.
    """
    result = {}
    for w in windows:
        result[f'hv_{w}d'] = returns.rolling(w).std() * np.sqrt(252)
    return pd.DataFrame(result)


from arch import arch_model

def garch_forecast(returns: pd.Series,
                   horizon: int = 5) -> dict:
    """
    Fit GARCH(1,1) and forecast volatility for next `horizon` days.
    Returns omega, alpha, beta parameters and the forecast.
    """
    # Scale returns to percent (arch library convention)
    r = returns * 100
    model = arch_model(r, vol='Garch', p=1, q=1, mean='constant')
    result = model.fit(disp='off')

    forecast = result.forecast(horizon=horizon)
    variance_forecast = forecast.variance.iloc[-1].values
    vol_forecast = np.sqrt(variance_forecast) / 100  # back to decimal

    return {
        'omega': round(float(result.params['omega']), 6),
        'alpha': round(float(result.params['alpha[1]']), 4),
        'beta': round(float(result.params['beta[1]']), 4),
        'persistence': round(float(result.params['alpha[1]'] +
                                  result.params['beta[1]']), 4),
        'vol_forecast_annualised': [
            round(float(v * np.sqrt(252)), 4) for v in vol_forecast
        ],
        'horizon_days': horizon
    }

