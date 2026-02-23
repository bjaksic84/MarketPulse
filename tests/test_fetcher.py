"""Tests for data fetcher and market config."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.data.market_config import (
    MarketConfig,
    load_market_config,
    load_strategy_config,
    list_available_markets,
    list_available_strategies,
)
from src.data.fetcher import YFinanceFetcher, create_fetcher


class TestMarketConfig:
    """Test market configuration loading and ticker formatting."""

    def test_load_stocks_config(self):
        config = load_market_config("stocks")
        assert config.name == "stocks"
        assert config.data_source == "yfinance"
        assert config.ticker_format == "{symbol}"
        assert "MSFT" in config.default_tickers
        assert config.has_splits is True
        assert config.use_adjusted_close is True

    def test_load_indices_config(self):
        config = load_market_config("indices")
        assert config.ticker_format == "^{symbol}"

    def test_format_ticker_stocks(self):
        config = load_market_config("stocks")
        assert config.format_ticker("MSFT") == "MSFT"
        assert config.format_ticker("NVDA") == "NVDA"

    def test_format_ticker_indices(self):
        config = load_market_config("indices")
        assert config.format_ticker("GSPC") == "^GSPC"
        assert config.format_ticker("DJI") == "^DJI"

    def test_format_tickers_list(self):
        config = load_market_config("indices")
        result = config.format_tickers(["GSPC", "DJI"])
        assert result == ["^GSPC", "^DJI"]

    def test_format_tickers_default(self):
        config = load_market_config("stocks")
        result = config.format_tickers()
        assert len(result) == len(config.default_tickers)
        assert result[0] == config.default_tickers[0]  # stocks: no transformation

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_market_config("nonexistent_market")

    def test_list_available_markets(self):
        markets = list_available_markets()
        assert "stocks" in markets
        assert "indices" in markets

    def test_load_strategy_config(self):
        config = load_strategy_config("short_term")
        assert config["name"] == "short_term"
        assert config["label_type"] == "classification"
        assert config["default_horizon"] == 5
        assert "model" in config
        assert "validation" in config

    def test_list_available_strategies(self):
        strategies = list_available_strategies()
        assert "short_term" in strategies


class TestYFinanceFetcher:
    """Test the YFinance data fetcher."""

    def test_create_fetcher(self):
        fetcher = create_fetcher(source="yfinance")
        assert isinstance(fetcher, YFinanceFetcher)

    def test_create_fetcher_invalid_source(self):
        with pytest.raises(ValueError, match="Unknown data source"):
            create_fetcher(source="nonexistent")

    def test_standardize_empty_df(self):
        fetcher = YFinanceFetcher()
        result = fetcher._standardize(pd.DataFrame(), "TEST")
        assert result.empty
        assert list(result.columns) == fetcher.OUTPUT_COLUMNS

    def test_standardize_renames_columns(self):
        fetcher = YFinanceFetcher()
        raw = pd.DataFrame({
            "Open": [100.0],
            "High": [105.0],
            "Low": [95.0],
            "Close": [102.0],
            "Volume": [1000000],
            "Adj Close": [102.0],
        }, index=pd.DatetimeIndex(["2024-01-01"]))

        result = fetcher._standardize(raw, "TEST")
        assert "open" in result.columns
        assert "close" in result.columns
        assert "adj_close" in result.columns
        assert result["open"].iloc[0] == 100.0

    def test_standardize_adds_adj_close_if_missing(self):
        fetcher = YFinanceFetcher()
        raw = pd.DataFrame({
            "Open": [100.0],
            "High": [105.0],
            "Low": [95.0],
            "Close": [102.0],
            "Volume": [1000000],
        }, index=pd.DatetimeIndex(["2024-01-01"]))

        result = fetcher._standardize(raw, "TEST")
        assert "adj_close" in result.columns
        assert result["adj_close"].iloc[0] == 102.0
