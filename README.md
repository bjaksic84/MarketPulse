# MarketPulse 📈

**AI-Driven Market Sentiment & Price Movement Predictor**

MarketPulse is a machine learning system that predicts price movements across multiple financial markets (stocks, crypto, futures, indices) using technical indicators, macro features, sentiment analysis, and ensemble tree models — supporting both classification and regression strategies.

## Features

- **Multi-Market Support:** Stocks, crypto, futures, indices — unified data pipeline via config-driven architecture
- **Multi-Strategy:** Short-term classification (3-day) and medium-term regression (5-10 day)
- **Ensemble Models:** XGBoost + LightGBM weighted soft-voting (classification & regression)
- **30+ Macro Features:** Calendar effects, VIX proxy, trend context, cross-asset signals
- **Market-Adaptive Features:** Per-market feature engineering (crypto weekends, futures sessions, etc.)
- **Regime Detection:** Volatility-based market regime identification
- **Technical Analysis:** 15+ indicators (RSI, MACD, Bollinger Bands, ATR, etc.) via pandas-ta
- **Explainable AI:** SHAP-based feature importance for every prediction
- **Stock Clustering:** K-Means / DBSCAN segmentation of stock behavior profiles
- **Walk-Forward Validation:** Time-series-safe evaluation — no look-ahead bias
- **Interactive Dashboard:** Streamlit app with regime overlays, regression views, ensemble predictions

## Project Structure

```
ML project/
├── config/                    # YAML configurations
│   ├── markets/               # Market-specific configs (stocks, crypto, futures...)
│   └── strategies/            # Strategy configs (short-term, medium-term...)
├── src/                       # Source code
│   ├── data/                  # Data fetching, preprocessing
│   ├── features/              # Feature engineering (technical, returns, labels)
│   ├── models/                # ML models (XGBoost, trainer, evaluator)
│   ├── analysis/              # Clustering & EDA
│   └── utils/                 # Validation, helpers
├── notebooks/                 # Jupyter notebooks for analysis
├── app/                       # Streamlit dashboard
└── tests/                     # Unit tests
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python -m src.models.trainer

# Launch the dashboard
streamlit run app/streamlit_app.py
```

## ML Methods Used

| Method | Application |
|---|---|
| **XGBoost Classifier** | Price direction classification (3-class) |
| **LightGBM Classifier** | Ensemble member for classification |
| **XGBoost Regressor** | Continuous return prediction |
| **LightGBM Regressor** | Ensemble member for regression |
| **Weighted Ensemble** | Soft-voting (classification) / weighted average (regression) |
| **K-Means / DBSCAN** | Stock behavior clustering |
| **SHAP** | Model explainability |
| **PCA** | Dimensionality reduction for cluster visualization |
| **Regime Detection** | Volatility-based market state identification |

## Data

- **Price data:** Yahoo Finance via yfinance (OHLCV, daily)
- **Universe:** S&P 500 large-caps + indices (SPY, ^VIX)
- **Timeframe:** 5 years of daily data (configurable)
- **Sentiment:** NewsAPI + FinBERT (Phase 2)

## Walk-Forward Validation

All models are evaluated using expanding-window walk-forward validation:
- Initial training window: 2 years
- Test window: 1 month (21 trading days)
- Step forward: 1 month
- No data leakage — features at time *t* only use data available at time *t*

## Roadmap

- [x] Phase 1: Core MVP (stocks, short-term, XGBoost)
- [x] Phase 2: Sentiment integration (NewsAPI + FinBERT)
- [x] Phase 3: Multi-market (crypto, indices, futures, LightGBM, ensemble, regime)
- [x] Phase 4: Multi-strategy (medium-term regression, macro features, dashboard)
- [ ] Phase 5: Production (scheduling, monitoring, backtesting)

## Test Coverage

**158 tests** across 7 test files:

| Phase | Tests | Coverage |
|---|---|---|
| Phase 1 | 55 | Data, features, labels, validation |
| Phase 2 | 18 | Sentiment pipeline |
| Phase 3 | 37 | Multi-market, regime, LightGBM, ensemble |
| Phase 4 | 31 | Macro features, regressors, regression evaluator |
| **Total** | **158** | All passing ✅ |

## License

MIT
