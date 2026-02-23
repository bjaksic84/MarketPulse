"""
Market configuration loader.

Reads YAML config files and returns structured MarketConfig objects.
Designed so that adding a new market = adding a new YAML file.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class MarketConfig:
    """Structured configuration for a single market."""

    name: str
    display_name: str
    data_source: str
    ticker_format: str
    calendar: str
    trading_days_per_year: int
    session: str
    default_tickers: List[str]
    benchmark: str
    volume_normalization: str
    has_splits: bool
    has_dividends: bool
    use_adjusted_close: bool
    has_gaps: bool
    feature_modules: List[str] = field(default_factory=list)

    def format_ticker(self, symbol: str) -> str:
        """Convert a base symbol to a yfinance-compatible ticker.

        Examples:
            stocks:  AAPL  -> AAPL
            crypto:  BTC   -> BTC-USD
            futures: ES    -> ES=F
            indices: GSPC  -> ^GSPC
        """
        return self.ticker_format.format(symbol=symbol)

    def format_tickers(self, symbols: Optional[List[str]] = None) -> List[str]:
        """Format a list of symbols (defaults to default_tickers)."""
        symbols = symbols or self.default_tickers
        return [self.format_ticker(s) for s in symbols]


# ──────────────────── Config Directory ────────────────────

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
MARKETS_DIR = CONFIG_DIR / "markets"
STRATEGIES_DIR = CONFIG_DIR / "strategies"


def load_market_config(market_name: str) -> MarketConfig:
    """Load a market config from YAML by name (e.g. 'stocks', 'crypto').

    Parameters
    ----------
    market_name : str
        Name of the market. Must match a YAML file in config/markets/.

    Returns
    -------
    MarketConfig
        Parsed configuration dataclass.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    """
    config_path = MARKETS_DIR / f"{market_name}.yaml"
    if not config_path.exists():
        available = [p.stem for p in MARKETS_DIR.glob("*.yaml")]
        raise FileNotFoundError(
            f"Market config '{market_name}' not found at {config_path}. "
            f"Available markets: {available}"
        )

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    m = raw["market"]
    return MarketConfig(
        name=m["name"],
        display_name=m["display_name"],
        data_source=m["data_source"],
        ticker_format=m["ticker_format"],
        calendar=m["calendar"],
        trading_days_per_year=m["trading_days_per_year"],
        session=m["session"],
        default_tickers=m["default_tickers"],
        benchmark=m["benchmark"],
        volume_normalization=m["volume_normalization"],
        has_splits=m["has_splits"],
        has_dividends=m["has_dividends"],
        use_adjusted_close=m["use_adjusted_close"],
        has_gaps=m["has_gaps"],
        feature_modules=m.get("feature_modules", []),
    )


def load_strategy_config(strategy_name: str) -> dict:
    """Load a strategy config from YAML by name (e.g. 'short_term').

    Returns the raw dict — strategy configs are more varied in structure
    and parsed by the consumer (trainer, evaluator, etc.).
    """
    config_path = STRATEGIES_DIR / f"{strategy_name}.yaml"
    if not config_path.exists():
        available = [p.stem for p in STRATEGIES_DIR.glob("*.yaml")]
        raise FileNotFoundError(
            f"Strategy config '{strategy_name}' not found at {config_path}. "
            f"Available strategies: {available}"
        )

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    return raw["strategy"]


def list_available_markets() -> List[str]:
    """Return names of all available market configs."""
    return sorted(p.stem for p in MARKETS_DIR.glob("*.yaml"))


def list_available_strategies() -> List[str]:
    """Return names of all available strategy configs."""
    return sorted(p.stem for p in STRATEGIES_DIR.glob("*.yaml"))
