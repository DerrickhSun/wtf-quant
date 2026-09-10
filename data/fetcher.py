import pandas as pd
import yfinance as yf


def fetch_prices(tickers: list[str] | str,
                 start: str,
                 end: str | None = None,
                 interval: str = '1d',
                 auto_adjust: bool = True) -> pd.DataFrame:
    """
    Close prices from Yahoo Finance.
    Single ticker -> Series-like single column named after the ticker.
    Multiple tickers -> one column per ticker, aligned on date.
    auto_adjust=False returns raw, unadjusted prices (for cross-checking
    against Yahoo's own history page, which shows Close and Adj Close
    as separate columns).
    """
    raw = yf.download(tickers, start=start, end=end, interval=interval,
                      auto_adjust=auto_adjust, progress=False)

    if raw.empty:
        raise ValueError(f'no data returned for {tickers}')

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw['Close']
    else:
        ticker = tickers if isinstance(tickers, str) else tickers[0]
        prices = raw[['Close']].rename(columns={'Close': ticker})

    return prices.dropna(how='all')


def fetch_ohlcv(ticker: str,
               start: str,
               end: str | None = None,
               interval: str = '1d',
               auto_adjust: bool = True) -> pd.DataFrame:
    """
    Full OHLCV history for a single ticker.
    auto_adjust=False returns raw, unadjusted prices (for cross-checking
    against Yahoo's own history page, which shows Close and Adj Close
    as separate columns).
    """
    raw = yf.download(ticker, start=start, end=end, interval=interval,
                      auto_adjust=auto_adjust, progress=False)

    if raw.empty:
        raise ValueError(f'no data returned for {ticker}')

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    return raw.dropna(how='all')
