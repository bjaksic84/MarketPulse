# MarketPulse — Technical Guide (Phases 1–4)

> **Version:** Phase 4 (Multi-Strategy + Medium-Term Regression)  
> **Updated:** February 2026

This document explains every component of the codebase: what each file does, the exact math behind each feature and model, how the YAML configs drive the system, and how to run tests and the full pipeline.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Architecture Overview](#2-architecture-overview)
3. [YAML Configuration System](#3-yaml-configuration-system)
4. [Data Layer](#4-data-layer)
5. [Feature Engineering](#5-feature-engineering)
6. [Label Generation](#6-label-generation)
7. [Walk-Forward Validation](#7-walk-forward-validation)
8. [Model — XGBoost Classifier](#8-model--xgboost-classifier)
9. [Evaluation System](#9-evaluation-system)
10. [Training Pipeline (Orchestrator)](#10-training-pipeline-orchestrator)
11. [Clustering Analysis](#11-clustering-analysis)
12. [Streamlit Dashboard](#12-streamlit-dashboard)
13. [Testing](#13-testing)
14. [Running the Project](#14-running-the-project)
15. [Key Design Principles](#15-key-design-principles)

---

## 1. Project Structure

```
ML project/
├── config/
│   ├── markets/
│   │   ├── stocks.yaml       # US Equities config
│   │   ├── crypto.yaml       # Cryptocurrency config
│   │   ├── futures.yaml      # Futures config
│   │   └── indices.yaml      # Market Indices config
│   └── strategies/
│       ├── short_term.yaml           # Short-term swing trading (classification)
│       ├── crypto_short_term.yaml    # Crypto-specific short-term
│       ├── futures_short_term.yaml   # Futures-specific short-term
│       ├── indices_short_term.yaml   # Indices-specific short-term
│       ├── medium_term.yaml          # Medium-term classification (5-10 day)
│       └── medium_term_regression.yaml  # Medium-term regression (continuous)
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── market_config.py  # YAML config loader + MarketConfig dataclass
│   │   ├── fetcher.py        # Data fetching abstraction (yfinance)
│   │   └── preprocessing.py  # OHLCV cleaning, normalization, returns
│   ├── features/
│   │   ├── __init__.py
│   │   ├── technical.py      # 30+ technical indicators (pandas-ta)
│   │   ├── returns.py        # Return-based statistical features
│   │   ├── labels.py         # Forward-looking target generation
│   │   ├── market_adaptive.py # Market-specific adaptive features (Phase 3)
│   │   └── macro.py          # Calendar, VIX proxy, trend context (Phase 4)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── xgboost_classifier.py   # XGBoost classifier + SHAP
│   │   ├── lightgbm_classifier.py  # LightGBM classifier (Phase 3)
│   │   ├── xgboost_regressor.py    # XGBoost regressor + SHAP (Phase 4)
│   │   ├── lightgbm_regressor.py   # LightGBM regressor (Phase 4)
│   │   ├── ensemble.py             # Weighted soft-voting ensemble (Phase 3+4)
│   │   ├── evaluator.py            # Classification metrics & reports
│   │   ├── regression_evaluator.py # Regression metrics & reports (Phase 4)
│   │   └── trainer.py              # Full training pipeline orchestrator
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── clustering.py     # K-Means / DBSCAN stock clustering
│   │   └── regime.py         # Market regime detection (Phase 3)
│   └── utils/
│       ├── __init__.py
│       └── validation.py     # Walk-forward time-series cross-validation
│
├── app/
│   └── streamlit_app.py      # Interactive dashboard (Phases 1-4 integrated)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_clustering.ipynb
│   ├── 05_sentiment_analysis.ipynb
│   ├── 06_multi_market.ipynb
│   ├── 07_ensemble_regime.ipynb
│   └── 08_multi_strategy.ipynb      # Phase 4 notebook
│
├── tests/
│   ├── __init__.py
│   ├── test_fetcher.py       # Config + fetcher tests (17 tests)
│   ├── test_features.py      # Technical + return feature tests (15 tests)
│   ├── test_labels.py        # Label generation + anti-leakage tests (11 tests)
│   ├── test_validation.py    # Walk-forward validation tests (12 tests)
│   ├── test_sentiment.py     # Sentiment pipeline tests (Phase 2)
│   ├── test_phase3.py        # Multi-market + ensemble tests (37 tests)
│   └── test_phase4.py        # Multi-strategy + regression tests (31 tests)
│
├── requirements.txt          # All Python dependencies
└── README.md                 # Project overview
```

---

## 2. Architecture Overview

The system follows a **pipeline architecture** where data flows linearly through discrete stages:

```
YAML Configs ──→ Data Fetcher ──→ Preprocessor ──→ Feature Engine ──→ Label Generator
                                                        │                     │
                                     Technical + Returns │          Classification
                                     Market-Adaptive     │          or Regression
                                     Macro Features      │                    │
              ┌───────────────────────────────────────────┘                    │
              ▼                                                               │
     Regime Detection ──→ Walk-Forward Splitter ──→ Ensemble/Model ──→ Evaluate
                              │                        │
                         Classification          Regression
                         (3-class)              (continuous)
                            │                       │
                        Evaluator          RegressionEvaluator
```

**Key architectural decisions:**

- **Config-driven**: Adding a new market = adding one YAML file. No code changes needed.
- **No look-ahead bias**: Labels are forward-looking (use future prices), features are backward-looking (use past prices only). These two sets of columns never mix.
- **Walk-forward validation**: No random shuffle. Training always precedes testing in time.
- **Modular**: Each `.py` file is self-contained with clear inputs/outputs.

---

## 3. YAML Configuration System

### How it works

`src/data/market_config.py` reads YAML files and converts them into Python objects:

- **`MarketConfig`** dataclass — holds all settings for one market.
- **`load_market_config(name)`** — reads `config/markets/{name}.yaml`, returns `MarketConfig`.
- **`load_strategy_config(name)`** — reads `config/strategies/{name}.yaml`, returns raw `dict`.

The config directory is resolved relative to the source file:
```python
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
```

### Market Configs (`config/markets/`)

Each market YAML defines how to fetch, format, and process data for that market.

| Field | Purpose | Example (stocks) |
|-------|---------|-------------------|
| `name` | Internal identifier | `"stocks"` |
| `display_name` | Human-readable name | `"US Equities"` |
| `data_source` | Which API to use | `"yfinance"` |
| `ticker_format` | How to build the yfinance symbol | `"{symbol}"` (stocks), `"{symbol}-USD"` (crypto), `"{symbol}=F"` (futures), `"^{symbol}"` (indices) |
| `calendar` | Trading calendar | `"NYSE"`, `"CME"`, `"24_7"` |
| `trading_days_per_year` | For annualization | 252 (stocks), 365 (crypto) |
| `default_tickers` | Universe of tickers to trade | `[AAPL, MSFT, GOOGL, ...]` |
| `benchmark` | For beta calculation | `"SPY"` |
| `volume_normalization` | Method for normalizing volume | `"z_score"`, `"min_max"`, or `"log"` |
| `use_adjusted_close` | Whether to adjust OHLC for splits/dividends | `true` (stocks), `false` (crypto) |
| `has_gaps` | Whether the market has non-trading days | `true` (stocks), `false` (crypto) |

**Ticker formatting** converts base symbols to yfinance-compatible tickers:

```python
MarketConfig.format_ticker("BTC")  # → "BTC-USD"  (crypto)
MarketConfig.format_ticker("ES")   # → "ES=F"     (futures)
MarketConfig.format_ticker("GSPC") # → "^GSPC"    (indices)
MarketConfig.format_ticker("AAPL") # → "AAPL"     (stocks)
```

### Strategy Config (`config/strategies/short_term.yaml`)

Defines the prediction task, model hyperparameters, and validation scheme.

| Section | Key Fields |
|---------|------------|
| **Prediction** | `horizon_days: [1, 3, 5]`, `default_horizon: 1`, `label_type: classification`, `num_classes: 3`, `threshold: 0.01` |
| **Model** | `type: xgboost_classifier`, `max_depth: 6`, `n_estimators: 300`, `learning_rate: 0.05`, `subsample: 0.8`, etc. |
| **Validation** | `method: walk_forward`, `initial_train_days: 504` (~2 years), `test_days: 21` (~1 month), `step_days: 21` |
| **Data** | `years_of_history: 5`, `min_data_points: 1000`, `max_missing_pct: 0.05` |
| **Feature Selection** | `max_features: 15`, `min_importance: 0.01` |

---

## 4. Data Layer

### 4.1 Fetcher (`src/data/fetcher.py`)

**What it does:** Downloads OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance.

**Classes:**
- `DataFetcher` — abstract base class defining the interface.
- `YFinanceFetcher` — concrete implementation using the `yfinance` library.
- `create_fetcher()` — factory function.

**Key behaviors:**

1. **Retry logic**: Up to 3 retries with exponential backoff (`delay * attempt` seconds).
2. **Column standardization**: Yahoo returns `Open, High, Low, Close, Volume, Adj Close` — we rename to lowercase: `open, high, low, close, volume, adj_close`.
3. **Batch download**: `fetch_multiple()` uses `yf.download()` with `group_by="ticker"` for efficiency.
4. **Error isolation**: If one ticker fails in a batch, others still return.

**Output schema** (every DataFrame has these columns):
```
open | high | low | close | volume | adj_close
```

### 4.2 Preprocessing (`src/data/preprocessing.py`)

**What it does:** Cleans raw OHLCV data and adds base derived columns.

**Pipeline steps (in order):**

1. **Drop duplicate dates** — keep last occurrence.
2. **Sort by date** — ensure chronological order.
3. **Quality check** — compare actual rows vs expected business days. Warn if >5% missing.
4. **Forward-fill gaps** — fill missing price data up to 3 consecutive days (configurable).
5. **Volume gap fill** — fill missing volume with 0.
6. **Adjusted close** — if `use_adjusted_close=true` (stocks), adjust OHLC columns:
   ```
   adj_ratio = adj_close / close
   open  = open  × adj_ratio
   high  = high  × adj_ratio
   low   = low   × adj_ratio
   close = adj_close
   ```
7. **Drop NaN close** — rows with no close price are removed.
8. **Compute returns:**
   - Simple return: $r_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1$
   - Log return: $r_t^{log} = \ln\left(\frac{P_t}{P_{t-1}}\right)$
9. **Normalize volume** — method determined by market config:
   - **Z-score** (default for stocks): $v_{norm} = \frac{V_t - \bar{V}_{50}}{\sigma_{V_{50}}}$ (50-day rolling window)
   - **Min-max**: $v_{norm} = \frac{V_t - V_{min,50}}{V_{max,50} - V_{min,50}}$
   - **Log**: $v_{norm} = \ln(1 + V_t)$
10. **Drop first row** — NaN from the returns calculation.

**Multi-ticker functions:**
- `preprocess_multiple()` — processes a dict of DataFrames, drops tickers with <50 clean rows.
- `align_to_common_dates()` — aligns multiple tickers to a common DatetimeIndex (inner or outer join).

---

## 5. Feature Engineering

All features are **backward-looking** — they use only past and current data. This is critical to avoid look-ahead bias.

### 5.1 Technical Indicators (`src/features/technical.py`)

Uses the `pandas-ta` library. Features are organized into 5 groups, each selectable via `include_groups` parameter.

#### Trend Indicators

| Feature | Formula / Description | Window |
|---------|----------------------|--------|
| `sma_20`, `sma_50`, `sma_200` | $SMA_n = \frac{1}{n}\sum_{i=0}^{n-1} P_{t-i}$ | 20, 50, 200 |
| `ema_12`, `ema_26` | $EMA_t = \alpha \cdot P_t + (1-\alpha) \cdot EMA_{t-1}$, where $\alpha = \frac{2}{n+1}$ | 12, 26 |
| `macd` | $MACD = EMA_{12} - EMA_{26}$ | 12, 26 |
| `macd_signal` | $Signal = EMA_9(MACD)$ | 9 |
| `macd_hist` | $Histogram = MACD - Signal$ | — |
| `adx` | Average Directional Index — measures trend strength (0-100). Calculated from DM+, DM-, and TR. | 14 |
| `dmp`, `dmn` | +DI and -DI (directional movement indicators) | 14 |

#### Momentum Indicators

| Feature | Formula / Description | Window |
|---------|----------------------|--------|
| `rsi_14` | $RSI = 100 - \frac{100}{1 + RS}$, where $RS = \frac{\text{avg gain}}{\text{avg loss}}$ | 14 |
| `stoch_k` | $\%K = \frac{P_t - L_{14}}{H_{14} - L_{14}} \times 100$ (where $L_{14}$, $H_{14}$ are 14-day low/high) | 14 |
| `stoch_d` | $\%D = SMA_3(\%K)$ | 3 |
| `cci_20` | $CCI = \frac{TP - SMA_{20}(TP)}{0.015 \times MAD_{20}(TP)}$, where $TP = \frac{H + L + C}{3}$ | 20 |
| `willr_14` | $\%R = \frac{H_{14} - P_t}{H_{14} - L_{14}} \times (-100)$ | 14 |

#### Volatility Indicators

| Feature | Formula / Description | Window |
|---------|----------------------|--------|
| `bb_upper` | $BB_{upper} = SMA_{20} + 2 \times \sigma_{20}$ | 20 |
| `bb_mid` | $BB_{mid} = SMA_{20}$ | 20 |
| `bb_lower` | $BB_{lower} = SMA_{20} - 2 \times \sigma_{20}$ | 20 |
| `bb_width` | $BW = \frac{BB_{upper} - BB_{lower}}{BB_{mid}}$ | — |
| `bb_pct` | $\%B = \frac{P_t - BB_{lower}}{BB_{upper} - BB_{lower}}$ | — |
| `atr_14` | $ATR = EMA_{14}(TR)$, where $TR = \max(H-L,\ |H-C_{t-1}|,\ |L-C_{t-1}|)$ | 14 |
| `atr_pct` | $ATR\% = \frac{ATR_{14}}{P_t}$ (normalized, comparable across assets) | — |

#### Volume Indicators

| Feature | Formula / Description | Window |
|---------|----------------------|--------|
| `obv` | On Balance Volume: $OBV_t = OBV_{t-1} + \text{sign}(r_t) \times V_t$ | cumulative |
| `vol_ratio_20` | $VR = \frac{V_t}{SMA_{20}(V)}$ (current volume vs 20-day avg) | 20 |

#### Custom / Cross-Indicator Features

| Feature | Formula | Type |
|---------|---------|------|
| `sma_cross_20_50` | $\mathbb{1}[SMA_{20} > SMA_{50}]$ | binary (0/1) |
| `sma_cross_50_200` | $\mathbb{1}[SMA_{50} > SMA_{200}]$ (Golden Cross signal) | binary |
| `dist_sma_20/50/200` | $\frac{P_t - SMA_n}{SMA_n}$ (% distance from moving average) | continuous |
| `rsi_oversold` | $\mathbb{1}[RSI < 30]$ | binary |
| `rsi_overbought` | $\mathbb{1}[RSI > 70]$ | binary |
| `macd_cross` | $\mathbb{1}[MACD > Signal]$ | binary |
| `trend_strong` | $\mathbb{1}[ADX > 25]$ | binary |
| `close_position` | $\frac{P_t - L_t}{H_t - L_t}$ (where in the daily range the close fell) | 0 to 1 |

**Total: ~33 technical indicator columns.**

### 5.2 Return Features (`src/features/returns.py`)

All computed from `close` and `returns` columns. Purely backward-looking.

#### Lagged Returns

$$r_{t,n} = \frac{P_t - P_{t-n}}{P_{t-n}}$$

Computed for $n \in \{1, 2, 3, 5, 10, 20\}$ days. These capture short-to-medium term price momentum.

#### Rolling Volatility

$$\sigma_{t,w} = \text{std}(r_{t-w+1}, \ldots, r_t)$$

Computed for windows $w \in \{5, 10, 20\}$ days.

#### Rolling Sharpe Ratio (Annualized)

$$Sharpe_{t,w} = \frac{\bar{r}_{t,w}}{\sigma_{t,w}} \times \sqrt{252}$$

where $\bar{r}_{t,w}$ is the rolling mean return over window $w$, and $\sigma_{t,w}$ is the rolling standard deviation. Computed for $w \in \{20, 60\}$.

#### Rolling Maximum Drawdown

$$MDD_{t,w} = \min_{i \in [t-w, t]} \left( \frac{P_i - \max_{j \in [t-w, i]} P_j}{\max_{j \in [t-w, i]} P_j} \right)$$

The worst peak-to-trough decline in the window. Always negative. Computed for $w \in \{20, 60\}$.

#### Momentum

$$Mom_{t,n} = \frac{P_t}{P_{t-n}} - 1$$

Same formula as lagged returns but semantically represents cumulative momentum. Computed for $n \in \{10, 20\}$.

#### Mean-Reversion Z-Score

$$z_{t,w} = \frac{P_t - \bar{P}_{t,w}}{\sigma_{P_{t,w}}}$$

How many standard deviations the current price is from its rolling mean. Computed for $w \in \{20, 50\}$. High positive values suggest overbought; high negative values suggest oversold.

#### Higher Moments (Rolling)

- **Skewness** ($w=20$): Asymmetry of return distribution. Negative skew = more large negative returns.
  $$\gamma_1 = \frac{\frac{1}{n}\sum(r_i - \bar{r})^3}{\sigma^3}$$
- **Kurtosis** ($w=20$): Fat-tailedness. High kurtosis = more extreme returns.
  $$\gamma_2 = \frac{\frac{1}{n}\sum(r_i - \bar{r})^4}{\sigma^4} - 3$$

#### Market Regime Features

| Feature | Formula |
|---------|---------|
| `up_ratio_20d` | $ \frac{1}{20}\sum_{i=0}^{19} \mathbb{1}[r_{t-i} > 0]$ (fraction of positive days) |
| `consec_up` | Count of consecutive positive return days ending at $t$ |
| `consec_down` | Count of consecutive negative return days ending at $t$ |

**Total: ~27 return feature columns.**

### 5.3 Combined Feature Count

The pipeline generates approximately **60 features** per ticker (33 technical + 27 return-based), though the exact number can vary slightly depending on data availability.

---

## 6. Label Generation (`src/features/labels.py`)

Labels represent the **future** price movement we're trying to predict.

### Forward Return

$$r_{t}^{fwd} = \frac{P_{t+h}}{P_t} - 1$$

where $h$ is the prediction horizon (number of trading days ahead).

### Classification Labels

**Binary (2-class):**
| Label | Condition | Numeric Code |
|-------|-----------|--------------|
| UP | $r^{fwd} \geq 0$ | 1 |
| DOWN | $r^{fwd} < 0$ | 0 |

**Ternary (3-class, default):**
| Label | Condition | Numeric Code |
|-------|-----------|--------------|
| UP | $r^{fwd} > +\theta$ | 2 |
| FLAT | $-\theta \leq r^{fwd} \leq +\theta$ | 1 |
| DOWN | $r^{fwd} < -\theta$ | 0 |

where $\theta$ is the threshold (default: 0.01 = ±1%).

### Anti-Look-Ahead-Bias Design

The function `get_clean_features_and_labels()` enforces strict separation:

- **Excluded from features (always):** `label`, `label_name`, `open`, `high`, `low`, `close`, `volume`, `adj_close`, and any column starting with `fwd_return`.
- The last $h$ rows have `NaN` labels (we don't know the future price yet) — these are dropped.
- Feature matrix `X` and label vector `y` are returned with aligned indices.

---

## 7. Walk-Forward Validation (`src/utils/validation.py`)

### Why not standard cross-validation?

Standard k-fold cross-validation **shuffles** data, which **leaks future information** into training. In time series, the model must only be trained on past data and tested on future data.

### How walk-forward works

```
Time ──────────────────────────────────────────────────────→

Fold 0: [====== TRAIN (504d) ======][TEST (21d)]
Fold 1: [======== TRAIN (525d) ========][TEST (21d)]
Fold 2: [========== TRAIN (546d) ==========][TEST (21d)]
  ...         (expanding window)
Fold N: [======================== TRAIN (1029d) ========================][TEST (21d)]
```

### Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `initial_train_days` | 504 | Minimum training window (~2 years of trading days) |
| `test_days` | 21 | Test window per fold (~1 month) |
| `step_days` | 21 | How far to step forward between folds |
| `expanding` | `true` | If true, training always starts at day 0 (window grows). If false, window slides (fixed size). |
| `min_train_samples` | 400 | Skip a fold if training set is smaller than this |

### Properties guaranteed by the implementation

1. **No overlap**: Training and test sets never share dates.
2. **Temporal ordering**: Every training sample is dated before every test sample within each fold.
3. **Expanding window**: Training set grows by `step_days` each fold (more data over time).
4. **No shuffling**: Data order is never randomized.

### Output

`split()` returns a list of `WalkForwardFold` dataclass instances, each containing:
- `train_start`, `train_end` (iloc positions)
- `test_start`, `test_end` (iloc positions)
- `train_start_date`, `test_end_date` (timestamps, if DatetimeIndex)

`get_fold_data(X, y, fold)` extracts the actual `(X_train, y_train, X_test, y_test)` DataFrames.

---

## 8. Model — XGBoost Classifier (`src/models/xgboost_classifier.py`)

### Why XGBoost?

XGBoost (eXtreme Gradient Boosting) is an ensemble of decision trees trained sequentially, where each tree corrects the errors of the previous ones. It satisfies the college requirement for "classification / trees."

### How XGBoost works (simplified)

1. Start with a base prediction (e.g., average class probability).
2. Compute the **residual errors** (gradient of the loss function).
3. Fit a new decision tree to predict these residuals.
4. Add the new tree's predictions (scaled by learning rate) to the ensemble.
5. Repeat for `n_estimators` rounds.

The loss function for multi-class is **softmax cross-entropy** (`multi:softprob`):

$$L = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \ln(p_{i,c})$$

where $p_{i,c}$ is the predicted probability of sample $i$ belonging to class $c$.

### Default Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `max_depth` | 6 | Maximum tree depth (controls complexity) |
| `n_estimators` | 300 | Number of boosting rounds |
| `learning_rate` | 0.05 | Shrinkage per tree (slower learning = better generalization) |
| `subsample` | 0.8 | Fraction of samples used per tree (stochastic) |
| `colsample_bytree` | 0.8 | Fraction of features used per tree |
| `min_child_weight` | 3 | Minimum sum of instance weight in a child (regularization) |
| `gamma` | 0.1 | Minimum loss reduction to make a split |
| `reg_alpha` | 0.1 | L1 regularization (lasso) |
| `reg_lambda` | 1.0 | L2 regularization (ridge) |
| `random_state` | 42 | Reproducibility seed |

### Class Balancing

Financial labels are often imbalanced (e.g., 55% FLAT, 25% UP, 21% DOWN). The model uses **balanced sample weights** via scikit-learn:

```python
sample_weight = compute_sample_weight("balanced", y_train)
```

This assigns higher weights to underrepresented classes so the model doesn't just predict the majority class.

### SHAP Explainability

After training, a `shap.TreeExplainer` is initialized. SHAP (SHapley Additive exPlanations) computes the contribution of each feature to each prediction:

- `get_feature_importance()` — returns XGBoost's built-in gain-based importance.
- `get_shap_values(X)` — computes per-sample, per-feature contribution values.
- Positive SHAP value = pushes prediction toward that class.
- Negative SHAP value = pushes prediction away from that class.

---

## 9. Evaluation System (`src/models/evaluator.py`)

### Per-Fold Metrics

Each walk-forward fold produces:

| Metric | Formula |
|--------|---------|
| **Accuracy** | $\frac{\text{correct predictions}}{\text{total predictions}}$ |
| **Precision** (weighted) | $\sum_c \frac{n_c}{N} \cdot \frac{TP_c}{TP_c + FP_c}$ |
| **Recall** (weighted) | $\sum_c \frac{n_c}{N} \cdot \frac{TP_c}{TP_c + FN_c}$ |
| **F1 Score** (weighted) | $\sum_c \frac{n_c}{N} \cdot \frac{2 \cdot Prec_c \cdot Rec_c}{Prec_c + Rec_c}$ |
| **ROC AUC** (weighted, OVR) | Area under the one-vs-rest ROC curve, weighted by class frequency |

### Aggregate Report

Across all folds:
- **Mean ± Std** of accuracy and F1 score.
- **Baseline accuracy** = majority class fraction (e.g., if 55% of labels are FLAT, a dummy model achieves 55%).
- **Lift** = model accuracy − baseline. Positive lift means the model beats random guessing.

### Visualizations

- `plot_fold_accuracy()` — line chart of accuracy per fold, with mean and baseline lines.
- `plot_confusion_matrix()` — heatmap of predicted vs actual across all folds.
- `plot_feature_importance()` — horizontal bar chart of top N features.

---

## 10. Training Pipeline (`src/models/trainer.py`)

`MarketPulseTrainer` is the **orchestrator** that connects everything.

### Pipeline steps for each ticker:

```
1. Load market config + strategy config
2. Fetch 5 years of daily OHLCV data (yfinance)
3. Preprocess (clean, adjust, normalize)
4. Compute ~33 technical indicator features
5. Compute ~27 return-based features
6. Generate forward-looking labels (3-class classification, ±1% threshold)
7. Extract clean features (X) and labels (y), drop NaN rows
8. Create walk-forward splits (26 folds for 5 years of data)
9. For each fold:
   a. Split into (X_train, y_train, X_test, y_test)
   b. Train XGBoost classifier
   c. Predict on test set
   d. Evaluate metrics
10. Aggregate results into EvaluationReport
11. Store the last trained model (for production predictions)
```

### CLI Usage

```bash
python -m src.models.trainer --market stocks --tickers AAPL MSFT --horizon 1
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--market` | `stocks` | Which market config to use |
| `--strategy` | `short_term` | Which strategy config to use |
| `--tickers` | From config | Space-separated base symbols |
| `--horizon` | From config (1) | Prediction horizon in days |

### `predict_latest()`

After training, call `predict_latest("AAPL")` to get the model's prediction for the next trading day:
```python
{
    "ticker": "AAPL",
    "prediction": "UP",        # or "DOWN" or "FLAT"
    "confidence": 0.62,        # max class probability
    "probabilities": {"DOWN": 0.15, "FLAT": 0.23, "UP": 0.62},
    "date": "2026-02-23"
}
```

---

## 11. Clustering Analysis (`src/analysis/clustering.py`)

This module satisfies the **college clustering requirement** in a financially meaningful way.

### Behavioral Features (per stock)

For each stock, 10 summary features are computed from its full price history:

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `mean_return` | $\bar{r} \times 252$ (annualized) | How fast the stock grows |
| `volatility` | $\sigma_r \times \sqrt{252}$ (annualized) | How much it fluctuates |
| `sharpe_ratio` | $\frac{\bar{r} \times 252}{\sigma_r \times \sqrt{252}}$ | Risk-adjusted return |
| `max_drawdown` | $\min_t \frac{P_t - \max_{s \leq t} P_s}{\max_{s \leq t} P_s}$ | Worst peak-to-trough drop |
| `avg_volume_usd` | $\log_{10}(\overline{V \times P})$ | Liquidity (log scale) |
| `beta` | $\frac{\text{Cov}(r_{stock}, r_{bench})}{\text{Var}(r_{bench})}$ | Market sensitivity |
| `skewness` | Third standardized moment of returns | Tail asymmetry |
| `kurtosis` | Fourth standardized moment − 3 (excess) | Tail fatness |
| `up_day_ratio` | $\frac{\text{count}(r > 0)}{N}$ | Consistency of gains |
| `avg_daily_range` | $\overline{\frac{H - L}{C}}$ | Intraday volatility proxy |

### K-Means Clustering

1. **Standardize** all 10 features (zero mean, unit variance) using `StandardScaler`.
2. **Elbow method**: Run K-Means for $K \in \{2, 3, 4, 5, 6, 7\}$.
   - Track **inertia** (within-cluster sum of squares): $\sum_{k} \sum_{x \in C_k} \|x - \mu_k\|^2$
   - Track **silhouette score**: $s = \frac{b - a}{\max(a, b)}$ where $a$ = avg intra-cluster distance, $b$ = avg nearest-cluster distance.
3. **Auto-select** optimal $K$ = the one with highest silhouette score.
4. **Final fit** with optimal $K$.

### DBSCAN (Comparison)

Density-based clustering as an alternative:
- `eps=1.5` — max distance between points in a cluster.
- `min_samples=2` — minimum points to form a dense region.
- Points not in any cluster are labeled as **noise** (-1).

### PCA Visualization

PCA reduces the 10 features to 2 dimensions for plotting:

$$Z = X_{std} \cdot W$$

where $W$ contains the first 2 eigenvectors of the covariance matrix. The resulting plot shows cluster separation in 2D, with explained variance percentages on axes.

### Cluster Interpretation

`interpret_clusters()` generates per-cluster profiles (mean of each feature) and auto-labels clusters:
- Volatility > 0.35 → "High-Volatility"
- Mean return > 0.15 → "Growth"
- Beta > 1.3 → "Aggressive"
- Sharpe > 1.0 → "Quality"
- etc.

---

## 12. Streamlit Dashboard (`app/streamlit_app.py`)

Interactive web dashboard with 4 views, accessible via:

```bash
streamlit run app/streamlit_app.py
```

### Views

| View | What it shows |
|------|---------------|
| **Ticker Explorer** | Candlestick chart with Bollinger Bands overlay, SMA 50/200, RSI subplot, volume bars. Select market, ticker, date range. |
| **Prediction** | Trains a model on-demand and shows: predicted direction (UP/FLAT/DOWN), confidence %, probability bar chart, top feature importances. |
| **Model Performance** | Full walk-forward evaluation with accuracy-over-time chart, aggregate metrics, confusion matrix. |
| **Clustering** | Runs clustering on selected stocks. Shows feature summary table, elbow plot, PCA scatter (interactive), cluster profiles. |

All data fetching and model training are cached with `@st.cache_data` to avoid re-downloading on every interaction.

---

## 13. Testing

### Test Suite Overview

55 tests across 4 files, all runnable with:

```bash
python -m pytest tests/ -v
```

### test_fetcher.py (17 tests)

Tests the configuration system and data fetching:

| Test | What it verifies |
|------|-----------------|
| `test_load_stocks_config` | Stocks YAML loads correctly, fields match |
| `test_load_crypto_config` | Crypto YAML loads, `trading_days_per_year=365` |
| `test_load_futures_config` | Futures YAML loads, CME calendar |
| `test_load_indices_config` | Indices YAML loads correctly |
| `test_format_ticker_*` (4 tests) | Ticker formatting: `AAPL→AAPL`, `BTC→BTC-USD`, `ES→ES=F`, `GSPC→^GSPC` |
| `test_default_tickers_not_empty` | Config has a non-empty ticker list |
| `test_missing_config` | `FileNotFoundError` for non-existent market |
| `test_strategy_config_loading` | Strategy config loads with expected keys |
| `test_strategy_validation_config` | Validation section has required fields |
| `test_strategy_model_config` | Model section has hyperparameters |
| `test_standardize_columns` | YFinanceFetcher renames columns correctly |
| `test_fetcher_empty_response` | Handles empty API responses gracefully |
| `test_create_fetcher_factory` | Factory function creates correct fetcher type |
| `test_format_tickers_batch` | Batch formatting of multiple tickers |

### test_features.py (15 tests)

Tests technical and return feature computation:

| Test | What it verifies |
|------|-----------------|
| `test_compute_indicators_adds_columns` | 30+ new columns added |
| `test_rsi_range` | RSI values are in [0, 100] |
| `test_sma_correctness` | SMA₂₀ matches manual rolling mean |
| `test_bollinger_band_ordering` | $BB_{lower} \leq BB_{mid} \leq BB_{upper}$ |
| `test_custom_features_binary` | Crossover signals are 0 or 1 |
| `test_selective_groups` | Can compute only specific indicator groups |
| `test_empty_dataframe` | Handles empty input without error |
| `test_return_features_*` (5 tests) | Lagged returns, volatility positivity, drawdown negativity, consecutive counts, etc. |

### test_labels.py (11 tests)

Tests label generation and the critical no-look-ahead-bias guarantee:

| Test | What it verifies |
|------|-----------------|
| `test_binary_classification` | 2 classes (0, 1) |
| `test_ternary_classification` | 3 classes (0, 1, 2) |
| `test_regression_labels` | Raw forward return as label |
| `test_horizon_nan_count` | Exactly `horizon` NaN labels at the end |
| `test_forward_return_correctness` | $r^{fwd} = P_{t+h}/P_t - 1$ matches manual calculation |
| **`test_no_look_ahead_bias`** | **Features at time $t$ do NOT contain any information from time $t+1$ or later** |
| `test_threshold_effect` | Higher threshold → more FLAT labels |
| `test_clean_features_extraction` | `get_clean_features_and_labels()` returns correct shapes |
| `test_label_columns_excluded` | Labels and price columns never appear in feature matrix |
| `test_index_alignment` | X.index == y.index |

### test_validation.py (12 tests)

Tests walk-forward validation logic:

| Test | What it verifies |
|------|-----------------|
| `test_split_generates_folds` | Produces >0 folds |
| `test_no_overlap` | Train and test index ranges don't intersect |
| `test_train_precedes_test` | `train_end < test_start` for every fold |
| `test_expanding_window` | Training set grows over time |
| `test_training_grows` | Each fold's train size ≥ previous fold's |
| `test_fold_data_extraction` | `get_fold_data()` returns correct shapes |
| `test_min_train_samples` | Folds below minimum are skipped |
| `test_sklearn_compatible` | `get_sklearn_splits()` returns correct format |
| `test_from_strategy_config` | Creates validator from YAML config |
| `test_fold_dates` | Date timestamps are present when index is DatetimeIndex |

### Running tests

```bash
# All tests
python -m pytest tests/ -v

# Specific file
python -m pytest tests/test_labels.py -v

# Specific test
python -m pytest tests/test_labels.py::test_no_look_ahead_bias -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

---

## 14. Running the Project

### Prerequisites

```bash
# Python 3.12 (3.14 is too new for some dependencies)
brew install python@3.12
brew install libomp  # Required by XGBoost on macOS

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Train a model

```bash
# Single ticker, 1-day horizon
python -m src.models.trainer --market stocks --tickers AAPL --horizon 1

# Multiple tickers
python -m src.models.trainer --market stocks --tickers AAPL MSFT GOOGL --horizon 5

# Crypto market
python -m src.models.trainer --market crypto --tickers BTC ETH --horizon 1

# Futures market
python -m src.models.trainer --market futures --tickers ES GC --horizon 3

# All default tickers in a market (14 stocks)
python -m src.models.trainer --market stocks --horizon 1
```

### Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

### Run tests

```bash
python -m pytest tests/ -v
```

---

## 15. Key Design Principles

### 1. No Look-Ahead Bias

The single most important rule in financial ML. At time $t$, the model can only use information available at time $t$ or earlier.

**How we enforce it:**
- Features are backward-looking (use past data).
- Labels are forward-looking (use future data).
- `get_clean_features_and_labels()` explicitly excludes `fwd_return_*`, `label`, and price columns from features.
- Walk-forward validation ensures training data is always earlier than test data.
- We have a dedicated test (`test_no_look_ahead_bias`) that verifies this.

### 2. Walk-Forward > k-Fold

Standard k-fold cross-validation shuffles data and assumes samples are i.i.d. (independent and identically distributed). Financial time series are **not i.i.d.** — they have autocorrelation, regime changes, and evolving distributions. Walk-forward validation respects this.

### 3. Config-Driven Extensibility

Everything market-specific or strategy-specific lives in YAML. To add crypto trading:
1. Config already exists (`config/markets/crypto.yaml`).
2. Run: `python -m src.models.trainer --market crypto --tickers BTC --horizon 1`.
3. No code changes.

### 4. Honest Evaluation

The baseline (majority class) is always shown alongside model accuracy. If the model doesn't beat the baseline, we report it honestly (negative lift). Phase 1 is the foundation — improving accuracy comes in later phases with feature selection, hyperparameter tuning, sentiment data, and ensemble methods.

### 5. Explainability

Every prediction has SHAP values attached. You can always ask "why did the model predict UP?" and get a ranked list of feature contributions. This is critical for:
- Building trust in predictions.
- Debugging model behavior.
- College presentations.

---

*This document covers Phase 1 (Core MVP). Future phases will add sentiment analysis (Phase 2), news integration (Phase 3), fundamental data (Phase 4), and production deployment (Phase 5+).*

---

## Phase 2: Sentiment Analysis & Model Tuning

### New Modules Added

#### `src/features/feature_selection.py`

Reduces the 60+ feature set to the most predictive subset.

**Methods:**

| Function | Description |
|---|---|
| `select_by_importance()` | Rank by XGBoost gain, keep top N |
| `select_by_mutual_info()` | Rank by mutual information with target label |
| `filter_correlated()` | Remove features with \|r\| > threshold, keep the more important one |
| `select_features_pipeline()` | Combined: correlation filter → ranking → top N |

**Math — Mutual Information:**

$$I(X; Y) = \sum_{y} \sum_{x} p(x, y) \log \frac{p(x, y)}{p(x) p(y)}$$

MI measures how much knowing feature $X$ reduces uncertainty about label $Y$. Unlike linear correlation, MI captures non-linear relationships.

**Correlation Filter Logic:**
For each pair $(f_i, f_j)$ with $|r_{ij}| > \theta$, drop the feature with lower XGBoost importance.

#### `src/models/tuner.py`

Hyperparameter optimization using walk-forward CV as the objective.

**Methods:**

| Method | Algorithm | Pros |
|---|---|---|
| `tune_optuna()` | Bayesian (TPE) | Most efficient, learns from previous trials |
| `tune_random()` | Random search | Simple baseline, no extra dependencies |

**Search Space (9 dimensions):**
- `max_depth` ∈ [3, 10], `n_estimators` ∈ [100, 800], `learning_rate` ∈ [0.01, 0.3]
- `subsample`, `colsample_bytree` ∈ [0.4, 1.0]
- `min_child_weight` ∈ [1, 10], `gamma` ∈ [0, 1]
- `reg_alpha`, `reg_lambda` ∈ [10⁻³, 10] (log-scale)

**CLI:**
```bash
python -m src.models.tuner --ticker AAPL --trials 30 --method optuna --folds 10
```

**Convenience function:**
```python
from src.models.tuner import quick_tune
best_params, score, trials_df = quick_tune("AAPL", n_trials=30)
```

#### `src/data/news_fetcher.py`

Multi-source news aggregator with 3 backends:

| Source | API Key? | History | Rate Limit |
|---|---|---|---|
| yfinance | No | ~Latest only | Unlimited |
| GDELT | No | 3 months | Unlimited |
| NewsAPI | Yes (free) | 1 month | 100 req/day |

All sources return standardized records with `title`, `source`, `published`, `url`, `ticker`.

**Usage:**
```python
from src.data.news_fetcher import AggregateNewsFetcher
fetcher = AggregateNewsFetcher(newsapi_key="optional")
news_df = fetcher.fetch_all("Apple", start_date="2024-01-01")
```

#### `src/features/sentiment.py`

FinBERT-powered sentiment scoring and daily feature aggregation.

**FinBERT Model:** `ProsusAI/finbert` — BERT fine-tuned on 60,000 financial sentences.

| Output | Score |
|---|---|
| Positive | +1.0 |
| Neutral | 0.0 |
| Negative | −1.0 |

**Daily Aggregated Features (8 total):**

| Feature | Description |
|---|---|
| `sentiment_mean` | Average headline sentiment (−1 to +1) |
| `sentiment_std` | Sentiment dispersion (disagreement among headlines) |
| `sentiment_count` | Number of headlines (news volume proxy) |
| `sentiment_positive_ratio` | Fraction of positive headlines |
| `sentiment_negative_ratio` | Fraction of negative headlines |
| `sentiment_momentum_3d` | 3-day rolling mean of daily sentiment |
| `sentiment_momentum_5d` | 5-day rolling mean of daily sentiment |
| `sentiment_surprise` | Today's sentiment − 5d rolling mean (contrarian signal) |

**Sentiment Surprise:**

$$\text{surprise}_t = s_t - \frac{1}{5}\sum_{i=1}^{5} s_{t-i}$$

A large positive surprise indicates unexpectedly bullish news → potential UP signal.

**CLI (sentiment-enabled training):**
```bash
python -m src.models.trainer --tickers AAPL --sentiment --newsapi-key YOUR_KEY
```

### Notebooks Added

| Notebook | Purpose |
|---|---|
| `05_tuning.ipynb` | Feature selection comparison, Optuna tuning, default vs tuned evaluation |
| `06_sentiment.ipynb` | News fetching, FinBERT scoring, daily aggregation, impact analysis |

### Test Coverage

Phase 2 adds 35 new tests (90 total):

| Test File | Tests | Covers |
|---|---|---|
| `test_feature_selection.py` | 17 | Correlation filter, importance selection, MI selection, pipeline |
| `test_sentiment.py` | 18 | Aggregation, merging, FinBERT (mocked), news fetcher structure |

---

# Phase 3 — Multi-Market, Ensemble & Regime Detection

> **Updated:** Phase 3

Phase 3 transforms MarketPulse from a stocks-only tool into a **multi-market prediction system** with market-adaptive features, ensemble modelling, and regime detection.

---

## Phase 3 New Files

```
config/strategies/
├── crypto_short_term.yaml      # Crypto-specific thresholds & params
├── futures_short_term.yaml     # Futures-specific strategy
├── indices_short_term.yaml     # Indices-specific strategy
└── short_term.yaml             # UPDATED: ensemble + market_adaptive sections

src/features/
└── market_adaptive.py          # Market-specific feature generators

src/models/
├── lightgbm_classifier.py      # LightGBM wrapper (MarketPulseLGBClassifier)
└── ensemble.py                 # Weighted soft-voting ensemble

src/analysis/
└── regime.py                   # Market regime detector (bull/neutral/bear)

notebooks/
└── 07_multi_market.ipynb       # Cross-market comparison & ensemble analysis

tests/
└── test_phase3.py              # 37 new tests for Phase 3 modules
```

---

## Market-Specific Strategy Configs

Each market gets a tuned YAML config reflecting its unique characteristics:

| Setting | Stocks | Crypto | Futures | Indices |
|---|---|---|---|---|
| **Threshold** | ±1.0% | ±2.0% | ±0.8% | ±0.5% |
| **Max Depth** | 6 | 8 | 5 | 5 |
| **N Estimators** | 300 | 400 | 250 | 350 |
| **Learning Rate** | 0.05 | 0.03 | 0.05 | 0.04 |
| **Initial Train** | 504d (~2y) | 365d (~1y) | 504d (~2y) | 504d (~2y) |
| **Test Window** | 21d | 14d | 21d | 21d |
| **Years History** | 5 | 4 | 5 | 8 |

**Why different?**
- **Crypto** has extreme volatility → wider threshold (±2%), deeper trees to capture non-linear patterns, shorter training window (pre-2020 data is a different market)
- **Futures** are leveraged → smaller threshold (±0.8%), shallower trees (cleaner signals), gap analysis for overnight settlement gaps
- **Indices** are diversified → smallest threshold (±0.5%), longer history captures mean-reversion patterns, VIX-proxy features

---

## Market-Adaptive Features

`src/features/market_adaptive.py` generates features tailored to each market type.

### Universal Features (all markets)

| Feature | Formula | Purpose |
|---|---|---|
| `vol_regime_ratio` | $\sigma_{10} / \sigma_{60}$ | Volatility expansion/contraction |
| `vol_expansion` | $\mathbb{1}[\text{ratio} > 1]$ | Binary vol expansion flag |
| `vol_regime_label` | Tercile of ratio (0/1/2) | Categorical regime |
| `volume_zscore` | $(V_t - \bar{V}_{20}) / \sigma_{V,20}$ | Standardised volume |
| `volume_spike` | $\mathbb{1}[V_t > k \cdot \bar{V}_{20}]$ | Volume spike flag |
| `volume_momentum_5d` | $\sum V_{5d} / \sum V_{20d}$ | Short-term volume momentum |

### Crypto-Only Features

| Feature | Formula | Purpose |
|---|---|---|
| `is_weekend` | $\mathbb{1}[\text{DOW} \geq 5]$ | Weekend trading flag (crypto is 24/7) |
| `day_of_week` | DOW / 6 (normalised) | Weekday seasonality |
| `extreme_up/down` | $\mathbb{1}[\|r_t\| > 5\%]$ | Extreme move indicators |
| `return_dispersion_5d` | $\max(r_{5d}) - \min(r_{5d})$ | Short-term return range |

### Futures-Only Features

| Feature | Formula | Purpose |
|---|---|---|
| `settlement_gap` | $O_t / C_{t-1} - 1$ | Overnight settlement gap |
| `is_month_end_week` | $\mathbb{1}[\text{day} \geq 25]$ | Contract roll proximity |
| `momentum_streak` | Consecutive same-sign returns | Trend persistence (futures trend more) |
| `trend_consistency_20d` | Fraction of days trending | How clean is the trend? |

### Index-Only Features

| Feature | Formula | Purpose |
|---|---|---|
| `reversion_zscore_{w}d` | $(P - \text{MA}_w) / \sigma_w$ for $w \in \{10,20,50\}$ | Mean-reversion signal |
| `dist_52w_high` | $(P - H_{252}) / H_{252}$ | Distance from 52-week high |
| `realized_vol_{10,30}d` | $\sigma_w \times \sqrt{252}$ | Annualised realised vol (VIX proxy) |
| `vol_term_structure` | $\text{vol}_{10d} / \text{vol}_{30d}$ | Short vs long vol (inverted = fear) |
| `days_since_big_move` | Count since $\|r_t\| > 1\%$ | Range compression → breakout signal |

### Gap Features (Stocks + Futures + Indices)

| Feature | Formula | Purpose |
|---|---|---|
| `overnight_gap` | $O_t / C_{t-1} - 1$ | Pre-market reaction |
| `gap_fill_ratio` | How much of the gap was closed intraday | Gap fill tendency |
| `gap_volatility_20d` | $\sigma_{\text{gap}, 20d}$ | Gap predictability |

---

## LightGBM Classifier

`src/models/lightgbm_classifier.py` — wraps LightGBM with the same API as the XGBoost classifier.

**Key differences from XGBoost:**
- **Leaf-wise growth** (vs depth-wise) → deeper, more specialised trees
- **Histogram-based binning** → faster training on large datasets
- **num_leaves=31** parameter controls tree complexity (instead of max_depth as primary)
- **Built-in categorical support** (future option)

API: `fit()`, `predict()`, `predict_proba()`, `get_feature_importance()`, `from_strategy_config()`

---

## Ensemble Model

`src/models/ensemble.py` — weighted soft-voting ensemble.

### Ensemble Prediction

$$P_{\text{ensemble}}(c) = \frac{\sum_{i} w_i \cdot P_i(c)}{\sum_{i} w_i}$$

Where $P_i(c)$ is model $i$'s probability for class $c$, and $w_i$ is the model weight.

### Agreement Score

When all models predict the same class, agreement = 1.0. Predictions with high agreement tend to be more reliable.

$$\text{agreement} = \frac{\max(\text{class\_counts})}{\text{num\_models}}$$

### Config

```yaml
ensemble:
  enabled: true
  models:
    - type: "xgboost_classifier"
      weight: 0.5
    - type: "lightgbm_classifier"
      weight: 0.5
```

---

## Market Regime Detector

`src/analysis/regime.py` — rules-based regime detection with no look-ahead bias.

### Three Regimes

| ID | Label | Characteristics |
|---|---|---|
| 0 | **Bearish** | Downtrend, high/rising vol |
| 1 | **Neutral** | Range-bound, moderate vol |
| 2 | **Bullish** | Uptrend, low/falling vol |

### Algorithm

1. **Trend direction**: Short EMA vs Long EMA → normalised to [-1, +1]
2. **Trend strength**: ADX-proxy = $|\bar{r}_w| / \overline{|r|}_w$ → [0, 1]
3. **Volatility regime**: $\sigma_{\text{short}} / \sigma_{\text{long}}$
4. **Combine**: $\text{score} = \text{dir} \times \text{strength} - \text{vol\_penalty}$
5. **Smooth**: Majority vote over a rolling window to avoid whipsaws

### Per-Market Tuning

| Setting | Stocks | Crypto | Futures | Indices |
|---|---|---|---|---|
| Short trend | 20 | 10 | 15 | 20 |
| Long trend | 50 | 30 | 40 | 50 |
| Smooth window | 5 | 3 | 5 | 7 |

Crypto uses shorter windows (faster regime shifts). Indices use longer smoothing (more stable regimes).

---

## Updated Training Pipeline

The trainer (`src/models/trainer.py`) now supports:

1. **`--no-ensemble`** flag to disable ensemble (use single XGBoost)
2. **`--no-adaptive`** flag to disable market-adaptive features
3. Automatic regime detection for all runs
4. Per-market strategy configs: `python -m src.models.trainer --market crypto --strategy crypto_short_term`

### Pipeline Flow (Phase 3+4)

```
Fetch → Preprocess → Technical Features → Return Features
    → Market-Adaptive Features (per market type)
    → Macro Features (calendar, VIX proxy, trend context)
    → Regime Detection
    → Labels (classification OR regression)
    → Walk-Forward Validation
        → Ensemble(XGBoost + LightGBM) or Single Model
        → Classification Evaluator OR Regression Evaluator
        → Aggregate Report
```

---

## Phase 3 Test Coverage

Phase 3 adds 37 new tests (127 total):

| Test Class | Tests | Covers |
|---|---|---|
| `TestVolatilityRegime` | 2 | Vol regime ratio, expansion flag, label |
| `TestGapFeatures` | 2 | Gap columns, direction flags |
| `TestVolumeSpike` | 2 | Spike detection, custom thresholds |
| `TestCryptoFeatures` | 2 | Weekend detection, extreme moves |
| `TestFuturesFeatures` | 2 | Session patterns, momentum streaks |
| `TestIndexFeatures` | 2 | Mean-reversion, 52w high distance |
| `TestAdaptiveDispatcher` | 5 | All 4 markets + feature listing |
| `TestRegimeConfig` | 2 | Defaults, per-market configs |
| `TestRegimeDetector` | 7 | Columns, values, bounds, summary, transitions |
| `TestLGBClassifier` | 5 | Fit/predict, proba, importance, binary |
| `TestEnsemble` | 6 | Fit/predict, agreement, individual preds, fallback |

---

## Phase 4: Multi-Strategy (Medium-Term Regression)

Phase 4 extends MarketPulse from a classification-only system to a **multi-strategy platform** supporting both classification and regression, with macro/calendar features and a fully integrated dashboard.

### 4.1 Macro Features (`src/features/macro.py`)

Three feature families (~30 features total):

#### Calendar Features (`compute_calendar_features`)
| Feature | Formula / Logic |
|---|---|
| `cal_day_of_week` | `dayofweek / 4` (normalized 0-1) |
| `cal_month_sin` | `sin(2π · month / 12)` |
| `cal_month_cos` | `cos(2π · month / 12)` |
| `cal_quarter` | `quarter / 4` |
| `cal_month_end` | Binary: last business day of month |
| `cal_quarter_end` | Binary: last business day of quarter |
| `cal_year_end` | Binary: last 5 business days of year |
| `cal_january` | Binary: January effect indicator |
| `cal_opex_week` | Binary: options expiry week (3rd Friday ±2 days) |
| `cal_is_monday` | Binary: Monday effect |
| `cal_is_friday` | Binary: Friday effect |

#### VIX Proxy Features (`compute_vix_proxy_features`)
Derived from the asset's own price history (no external VIX data needed):

| Feature | Formula |
|---|---|
| `vix_proxy_5d/10d/20d` | `std(log_returns, window) × √252 × 100` |
| `vix_term_structure` | `vix_proxy_5d / vix_proxy_20d` (mean-reversion signal) |
| `vix_percentile` | Expanding percentile rank of 20d vol |
| `vix_spike` | Binary: `vix_proxy_5d > 1.5 × vix_proxy_20d` |
| `vix_high_fear` | Binary: `vix_percentile > 0.8` |
| `vix_low_fear` | Binary: `vix_percentile < 0.2` |

#### Trend Context Features (`compute_trend_context_features`)
| Feature | Formula |
|---|---|
| `dist_ma_50d/100d/200d` | `(Close - MA) / MA` (distance from moving average) |
| `golden_cross` | Binary: 50d MA > 200d MA |
| `cum_return_20d/60d/120d` | Cumulative log return over window |
| `drawdown` | `(Close - cummax) / cummax` |
| `drawdown_recovery` | Binary: current drawdown > -1% (recovering) |
| `higher_high_20d` | Binary: Close > 20d rolling max |
| `lower_low_20d` | Binary: Close < 20d rolling min |

### 4.2 XGBoost Regressor (`src/models/xgboost_regressor.py`)

`MarketPulseXGBRegressor` wraps `xgboost.XGBRegressor` with the same API as the classifier:

- **Objective:** `reg:squarederror`
- **`from_strategy_config(cfg)`**: Builds from YAML (max_depth, n_estimators, learning_rate, reg_lambda, etc.)
- **`predict_proba(X)`**: Returns `(n, 1)` array (raw predictions) for ensemble API compatibility
- **`get_shap_values(X)`**: SHAP TreeExplainer for feature attribution
- **`get_feature_importance()`**: Built-in XGBoost gain-based importance

### 4.3 LightGBM Regressor (`src/models/lightgbm_regressor.py`)

`MarketPulseLGBRegressor` wraps `lightgbm.LGBMRegressor`:

- **Objective:** `regression` (MSE)
- **`fit(**kwargs)`**: Gracefully ignores `balance_classes` kwarg (classification-only parameter)
- Same API surface as XGBoost regressor

### 4.4 Regression Evaluator (`src/models/regression_evaluator.py`)

`RegressionEvaluator` mirrors the classification `Evaluator` but for continuous targets:

| Metric | Formula |
|---|---|
| MAE | `mean(\|y_true - y_pred\|)` |
| RMSE | `√(mean((y_true - y_pred)²))` |
| R² | `1 - SS_res / SS_tot` |
| Directional Accuracy | `mean(sign(y_true) == sign(y_pred))` |
| Information Coefficient | Spearman rank correlation between predictions and actuals |

**Dataclasses:** `RegressionFoldResult` (per-fold) and `RegressionReport` (aggregated).

### 4.5 Ensemble Regression Support

The `MarketPulseEnsemble` (Phase 3) now auto-detects regression mode:

```python
self._is_regression = any(
    name in ("xgboost_regressor", "lightgbm_regressor")
    for name, _ in self.model_weights
)
```

- **Classification mode**: Weighted soft-voting via `predict_proba()` → `argmax`
- **Regression mode**: Weighted average of raw predictions → continuous output
- `predict_proba()` returns `(n, 1)` in regression mode

### 4.6 Strategy Configs

#### `medium_term.yaml` (Classification)
- **Horizon:** 5-10 days (vs 3d for short_term)
- **Threshold:** ±2% (larger moves for medium-term)
- **Features:** technical + returns + market_adaptive + macro + sentiment
- **Ensemble:** XGBoost + LightGBM classifiers (50/50)
- **max_depth:** 5, **n_estimators:** 400, **learning_rate:** 0.03
- **mean_reversion:** true (trend context features enabled)

#### `medium_term_regression.yaml` (Regression)
- **label_type:** `regression` (continuous 5-day forward return)
- **Model:** xgboost_regressor (primary)
- **Ensemble:** xgboost_regressor + lightgbm_regressor (50/50)
- **reg_lambda:** 2.0 (stronger regularization for regression)
- **years_of_history:** 6, **max_features:** 25

### 4.7 Dashboard Integration

The Streamlit dashboard (`app/streamlit_app.py`) was fully rewritten to integrate all Phase 3+4 features:

- **Strategy selector**: Choose between all available strategies per market
- **Ensemble toggle**: Enable/disable weighted soft-voting
- **Enriched data pipeline**: Technical → Returns → Adaptive → Regime → Macro
- **Regime overlay**: Visual regime detection on price charts
- **Regression view**: Predicted vs actual scatter plots, directional accuracy per fold, full regression metrics
- **4-column prediction display**: Prediction, confidence %, horizon, agreement score

### 4.8 Pipeline Updates

The trainer (`src/models/trainer.py`) now:
1. Computes macro features when `macro` is in the strategy's feature list
2. Supports `label_type: regression` for continuous target generation
3. Uses `RegressionEvaluator` when in regression mode
4. Passes macro features through `predict_latest()` for live predictions

---

## Phase 4 Test Coverage

Phase 4 adds 31 new tests (158 total):

| Test Class | Tests | Covers |
|---|---|---|
| `TestMacroFeatures` | 6 | Calendar, VIX proxy, trend context, full pipeline, names, NaN tail |
| `TestXGBRegressor` | 5 | Fit/predict, proba shape, importance, config, SHAP |
| `TestLGBRegressor` | 4 | Fit/predict, proba shape, importance, config |
| `TestRegressionEvaluator` | 5 | Fold eval, directional accuracy, IC, aggregation, report |
| `TestEnsembleRegression` | 6 | Regression detect, classification detect, fit/predict, proba, importance, classification compat |
| `TestStrategyConfigs` | 2 | medium_term loads, medium_term_regression loads |
| `TestInitExports` | 2 | Models __init__, features __init__ |
| `TestDashboardImports` | 1 | All dashboard imports resolve |
