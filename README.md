# MarketPulse 📈

**AI-Driven Market Price Movement Predictor**

MarketPulse is a machine learning system that predicts price movements for **US indices** (primary) and **US stocks** (secondary) using technical indicators, macro features, sentiment analysis, and ensemble tree models — supporting both classification and regression strategies.

## Features

- **Indices + Stocks Focus:** US indices (S&P 500, Dow, Nasdaq, Russell 2000) as primary market, curated stock universe as secondary
- **Binary Classification:** 2-class UP/DOWN prediction with honest threshold calibration
- **Multi-Strategy:** Short-term (1-5 day) and medium-term (5-10 day) horizons
- **Ensemble Models:** XGBoost + LightGBM weighted soft-voting (0.7/0.3) with ensemble-level calibration
- **36 Macro Features:** Calendar effects, VIX proxy, trend context, cross-asset signals
- **Market-Adaptive Features:** Per-market feature engineering (index mean-reversion, gap analysis, volatility regime)
- **Feature Selection:** Top-8 features by importance, correlation threshold 0.85
- **Regime Detection:** Volatility-based market regime identification (bull / neutral / bear)
- **Technical Analysis:** 15+ normalized indicators (RSI, MACD, Bollinger Bands, ATR, etc.) via pandas-ta
- **Explainable AI:** SHAP-based feature importance for every prediction
- **Stock Clustering:** K-Means / DBSCAN segmentation of stock behavior profiles
- **Walk-Forward Validation:** Time-series-safe evaluation with purge gap — no look-ahead bias
- **Interactive Dashboard:** Streamlit app with regime overlays, ensemble predictions

## Accuracy

Validated via walk-forward cross-validation (expanding window, 42-day test, 21-day step):

| Market | Representative Tickers | Avg Accuracy |
|---|---|---|
| **Indices** | ^GSPC, ^DJI, ^IXIC, ^RUT | **53.7%** |
| **Stocks** | MSFT, TSLA, NVDA, META, JPM, GS, V, MA, SPY | **51.2%** |
| **Overall** | 13 tickers | **~52%** (>50% beats random) |

## Project Structure

```
ML project/
├── config/                    # YAML configurations
│   ├── markets/               # Market configs (stocks.yaml, indices.yaml)
│   └── strategies/            # Strategy configs (short_term, medium_term, indices_*)
├── src/                       # Source code
│   ├── data/                  # Data fetching, preprocessing
│   ├── features/              # Feature engineering (technical, returns, macro, labels)
│   ├── models/                # ML models (XGBoost, LightGBM, ensemble, trainer)
│   ├── analysis/              # Clustering & regime detection
│   └── utils/                 # Walk-forward validation
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
| **XGBoost Classifier** | Binary price direction classification |
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
- **Index universe:** ^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (Nasdaq), ^RUT (Russell 2000)
- **Stock universe:** MSFT, TSLA, NVDA, META, JPM, GS, V, MA, SPY
- **Timeframe:** 5 years of daily data (configurable)
- **Sentiment:** NewsAPI + FinBERT (Phase 2)

## Walk-Forward Validation

All models are evaluated using expanding-window walk-forward validation:
- Initial training window: 2 years (504 trading days)
- Test window: 42 trading days (~2 months)
- Step forward: 21 trading days (~1 month)
- Purge gap: horizon days between train and test (prevents label leakage)
- No data leakage — features at time *t* only use data available at time *t*

## Roadmap

- [x] Phase 1: Core MVP (stocks, short-term, XGBoost)
- [x] Phase 2: Sentiment integration (NewsAPI + FinBERT)
- [x] Phase 3: Multi-market support (indices + stocks, LightGBM, ensemble, regime)
- [x] Phase 4: Multi-strategy (medium-term regression, macro features, dashboard)
- [x] Phase 5: Accuracy optimization (binary classification, feature normalization, honest calibration, market focus)
- [ ] Phase 6: Production (scheduling, monitoring, backtesting)

## Test Coverage

**148 tests** across 7 test files:

| Phase | Tests | Coverage |
|---|---|---|
| Phase 1 | 52 | Data, features, labels, validation |
| Phase 2 | 18 | Sentiment pipeline |
| Phase 3 | 31 | Multi-market, regime, LightGBM, ensemble |
| Phase 4 | 31 | Macro features, regressors, regression evaluator |
| Phase 5 | 16 | Feature selection, accuracy diagnostics |
| **Total** | **148** | All passing ✅ |

## License

MIT
