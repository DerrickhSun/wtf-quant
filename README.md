# wtf-quant

Repository structure:
wtf-quant/
  models/
    risk.py             — VaR, CVaR, Sharpe, drawdown
    volatility.py       — historical vol, GARCH(1,1)
    monte_carlo.py      — GBM simulation, options, portfolio
    black_scholes.py    — analytical BS formula (validation benchmark)
  data/
    fetcher.py          — yfinance price data fetcher
    cache.py            — local price cache
  api/
    main.py             — FastAPI app
    routes.py           — endpoint handlers
    schemas.py          — Pydantic request/response models
  notebooks/
    01_risk_metrics.ipynb
    02_volatility_modelling.ipynb
    03_monte_carlo.ipynb
  tests/
    test_risk.py        — VaR/CVaR unit tests
    test_volatility.py
    test_monte_carlo.py
    test_bs_parity.py   — MC options vs Black-Scholes convergence
  Dockerfile
  requirements.txt
  README.md
