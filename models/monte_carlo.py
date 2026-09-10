def simulate_gbm(S0: float,
                  mu: float,
                  sigma: float,
                  T_days: int = 252,
                  n_paths: int = 10000,
                  seed: int = 42) -> np.ndarray:
    """
    Simulate n_paths GBM price paths over T_days trading days.
    Returns shape (n_paths, T_days+1).
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    # Antithetic variates for variance reduction
    Z = rng.standard_normal((n_paths // 2, T_days))
    Z = np.vstack([Z, -Z])

    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    paths = S0 * np.exp(np.cumsum(log_returns, axis=1))
    return np.hstack([np.full((n_paths, 1), S0), paths])

def monte_carlo_option(S0: float, K: float, r: float,
                        sigma: float, T_years: float,
                        option_type: str = 'call',
                        n_paths: int = 100000) -> dict:
    """
    Price a European option via Monte Carlo.
    Also returns the Black-Scholes analytical price for comparison.
    """
    T_days = int(T_years * 252)
    paths = simulate_gbm(S0, r, sigma, T_days, n_paths)
    ST = paths[:, -1]  # terminal prices

    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    mc_price = float(np.exp(-r * T_years) * np.mean(payoffs))
    mc_stderr = float(np.std(payoffs) / np.sqrt(n_paths))

    # Black-Scholes analytical price for validation
    bs_price = black_scholes(S0, K, r, sigma, T_years, option_type)

    return {
        'mc_price': round(mc_price, 4),
        'mc_stderr': round(mc_stderr, 4),
        'bs_price': round(bs_price, 4),
        'mc_bs_diff': round(abs(mc_price - bs_price), 4),
        'n_paths': n_paths
    }

def portfolio_simulation(weights: np.ndarray,
                          mu_vec: np.ndarray,
                          cov_matrix: np.ndarray,
                          T_days: int = 252,
                          n_paths: int = 10000) -> dict:
    """
    Simulate a weighted portfolio using correlated GBM paths.
    Uses Cholesky decomposition to generate correlated returns.
    """
    n_assets = len(weights)
    L = np.linalg.cholesky(cov_matrix)  # Cholesky decomposition

    rng = np.random.default_rng(42)
    Z = rng.standard_normal((n_paths, T_days, n_assets))
    correlated_Z = Z @ L.T

    dt = 1 / 252
    log_returns = ((mu_vec - 0.5 * np.diag(cov_matrix)) * dt +
                    np.sqrt(dt) * correlated_Z)

    asset_paths = np.exp(np.cumsum(log_returns, axis=1))
    portfolio_paths = (asset_paths * weights).sum(axis=2)
    final_returns = portfolio_paths[:, -1] - 1

    return {
        'mean_return': round(float(final_returns.mean()), 4),
        'std_return': round(float(final_returns.std()), 4),
        'var_95': round(float(np.percentile(final_returns, 5)), 4),
        'cvar_95': round(float(final_returns[final_returns <=
                    np.percentile(final_returns, 5)].mean()), 4),
        'prob_profit': round(float((final_returns > 0).mean()), 4),
        'percentile_10': round(float(np.percentile(final_returns, 10)), 4),
        'percentile_90': round(float(np.percentile(final_returns, 90)), 4),
    }
