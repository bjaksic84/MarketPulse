# MarketPulse 📈

**AI-Driven Market Sentiment & Price Movement Predictor**

MarketPulse is a machine learning system that predicts short-term price movements across multiple financial markets (stocks, crypto, futures, indices) using technical indicators, sentiment analysis, and ensemble tree models.

## Features

- **Multi-Market Support:** Stocks, crypto, futures, indices — unified data pipeline via config-driven architecture
- **Price Movement Prediction:** XGBoost classifier predicting next-day direction (UP / DOWN / FLAT)
- **Technical Analysis:** 15+ indicators (RSI, MACD, Bollinger Bands, ATR, etc.) via pandas-ta
- **Explainable AI:** SHAP-based feature importance for every prediction
- **Stock Clustering:** K-Means / DBSCAN segmentation of stock behavior profiles
- **Walk-Forward Validation:** Time-series-safe evaluation — no look-ahead bias
- **Interactive Dashboard:** Streamlit app with candlestick charts, predictions, and cluster visualizations

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
| **XGBoost / Random Forest** | Price direction classification |
| **Logistic Regression** | Interpretable baseline |
| **K-Means / DBSCAN** | Stock behavior clustering |
| **Linear Regression** | Return magnitude baseline |
| **SHAP** | Model explainability |
| **PCA** | Dimensionality reduction for cluster visualization |

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
- [ ] Phase 2: Sentiment integration (NewsAPI + FinBERT)
- [ ] Phase 3: Multi-market (crypto, indices, futures)
- [ ] Phase 4: Multi-strategy (medium-term regression)
- [ ] Phase 5: Regime detection (HMM, long-term clustering)
- [ ] Phase 6: Production (scheduling, monitoring, backtesting)

## License

MIT
