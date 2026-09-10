from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from data.fetcher import fetch_ohlcv, fetch_prices

DATES = pd.date_range('2024-01-02', periods=5, freq='B')


def _single_ticker_frame() -> pd.DataFrame:
    return pd.DataFrame({
        'Open': np.arange(5) + 100.0,
        'High': np.arange(5) + 101.0,
        'Low': np.arange(5) + 99.0,
        'Close': np.arange(5) + 100.5,
        'Volume': np.arange(5) * 1000,
    }, index=DATES)


def _multi_ticker_frame(tickers: list[str]) -> pd.DataFrame:
    fields = ['Open', 'High', 'Low', 'Close', 'Volume']
    columns = pd.MultiIndex.from_product([fields, tickers])
    data = np.arange(len(DATES) * len(fields) * len(tickers)).reshape(
        len(DATES), len(fields) * len(tickers))
    return pd.DataFrame(data, index=DATES, columns=columns)


@patch('data.fetcher.yf.download')
def test_fetch_prices_single_ticker(mock_download):
    mock_download.return_value = _single_ticker_frame()

    result = fetch_prices('AAPL', start='2024-01-01', end='2024-01-08')

    assert list(result.columns) == ['AAPL']
    assert len(result) == 5


@patch('data.fetcher.yf.download')
def test_fetch_prices_multiple_tickers(mock_download):
    tickers = ['AAPL', 'MSFT']
    mock_download.return_value = _multi_ticker_frame(tickers)

    result = fetch_prices(tickers, start='2024-01-01', end='2024-01-08')

    assert set(result.columns) == set(tickers)
    assert len(result) == 5


@patch('data.fetcher.yf.download')
def test_fetch_prices_empty_raises(mock_download):
    mock_download.return_value = pd.DataFrame()

    with pytest.raises(ValueError):
        fetch_prices('AAPL', start='2024-01-01', end='2024-01-08')


@patch('data.fetcher.yf.download')
def test_fetch_ohlcv_single_ticker(mock_download):
    mock_download.return_value = _single_ticker_frame()

    result = fetch_ohlcv('AAPL', start='2024-01-01', end='2024-01-08')

    assert set(['Open', 'High', 'Low', 'Close', 'Volume']) <= set(result.columns)
    assert len(result) == 5


@patch('data.fetcher.yf.download')
def test_fetch_ohlcv_flattens_multiindex(mock_download):
    # yfinance can return MultiIndex columns for a single ticker too,
    # depending on the group_by setting.
    mock_download.return_value = _multi_ticker_frame(['AAPL'])

    result = fetch_ohlcv('AAPL', start='2024-01-01', end='2024-01-08')

    assert not isinstance(result.columns, pd.MultiIndex)
    assert set(['Open', 'High', 'Low', 'Close', 'Volume']) <= set(result.columns)


@patch('data.fetcher.yf.download')
def test_fetch_ohlcv_empty_raises(mock_download):
    mock_download.return_value = pd.DataFrame()

    with pytest.raises(ValueError):
        fetch_ohlcv('AAPL', start='2024-01-01', end='2024-01-08')
