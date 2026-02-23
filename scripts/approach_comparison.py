#!/usr/bin/env python3
"""
Test different approaches to find the best accuracy:
1. Curated minimal features (hand-picked, theory-driven)
2. Very shallow models (max_depth 2)
3. Single models vs ensemble
4. Different feature counts
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data.fetcher import YFinanceFetcher
from src.data.market_config import load_market_config, load_strategy_config
from src.data.preprocessing import preprocess_ohlcv
from src.features.technical import compute_technical_indicators
from src.features.returns import compute_return_features
from src.features.labels import generate_labels, get_clean_features_and_labels
from src.analysis.regime import detect_regime
from src.features.market_adaptive import compute_market_adaptive_features
from src.models.xgboost_classifier import MarketPulseXGBClassifier
from src.models.ensemble import MarketPulseEnsemble
from src.utils.validation import WalkForwardValidator
from src.features.feature_selection import select_features_pipeline


# Curated feature sets (theory-driven, minimal)
CURATED_FEATURES_MINIMAL = [
    # Momentum (strongest signal in time series)
    "ret_5d", "ret_10d", "ret_20d", "momentum_20d",
    # Mean reversion
    "zscore_20d", "bb_pct",
    # Volatility state
    "atr_pct", "vol_10d", "vol_regime_ratio",
    # Trend
    "adx", "rsi_14", "dist_sma_50_pct",
]

CURATED_FEATURES_EXTENDED = CURATED_FEATURES_MINIMAL + [
    # Volume
    "vol_ratio_20", "volume_zscore",
    # More momentum/mean reversion
    "sharpe_20d", "sharpe_60d", "max_dd_20d",
    # Trend signals
    "sma_cross_50_200", "macd_cross", "trend_direction",
]


def build_data(ticker, market_name, strategy_name, horizon=5, num_classes=2):
    market_config = load_market_config(market_name)
    strategy_config = load_strategy_config(strategy_name)
    years = strategy_config.get("data", {}).get("years_of_history", 5)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')

    fetcher = YFinanceFetcher(market_config=market_config)
    raw = fetcher.fetch(ticker, start=start_date, end=end_date)
    df = preprocess_ohlcv(raw, market_config=market_config)
    df = compute_technical_indicators(df)
    df = compute_return_features(df)
    df = compute_market_adaptive_features(df, market_name=market_name, strategy_config=strategy_config)
    df = detect_regime(df, market_name=market_name)
    df = generate_labels(df, horizon=horizon, label_type='classification',
                         num_classes=num_classes, threshold=0.0)
    X_all, y = get_clean_features_and_labels(df)
    return X_all, y, strategy_config


def run_wf_test(X, y, model_fn, init_train=504, test_days=42, step_days=21, purge=5):
    validator = WalkForwardValidator(
        initial_train_days=init_train,
        test_days=test_days,
        step_days=step_days,
        purge_days=purge,
    )
    folds = validator.split(X)
    if not folds:
        return None
    all_true, all_pred, fold_accs = [], [], []
    for fold in folds:
        X_train = X.iloc[fold.train_start:fold.train_end+1]
        y_train = y.iloc[fold.train_start:fold.train_end+1]
        X_test = X.iloc[fold.test_start:fold.test_end+1]
        y_test = y.iloc[fold.test_start:fold.test_end+1]
        y_pred = model_fn(X_train, y_train, X_test)
        acc = accuracy_score(y_test.values.astype(int), y_pred)
        fold_accs.append(acc)
        all_true.extend(y_test.values.astype(int))
        all_pred.extend(y_pred)
    mean_acc = np.mean(fold_accs)
    baseline = pd.Series(all_true).value_counts(normalize=True).max()
    return mean_acc, baseline, mean_acc - baseline, len(folds)


def xgb_model(max_depth=2, n_est=100, lr=0.02, num_classes=2):
    def fn(X_tr, y_tr, X_te):
        from xgboost import XGBClassifier
        from sklearn.utils.class_weight import compute_sample_weight
        obj = "binary:logistic" if num_classes == 2 else "multi:softprob"
        m = XGBClassifier(
            max_depth=max_depth, n_estimators=n_est, learning_rate=lr,
            subsample=0.7, colsample_bytree=0.6, min_child_weight=10,
            gamma=0.5, reg_alpha=0.5, reg_lambda=3.0,
            objective=obj, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )
        sw = compute_sample_weight("balanced", y_tr.astype(int))
        m.fit(X_tr, y_tr.astype(int), sample_weight=sw, verbose=False)
        return m.predict(X_te)
    return fn


def logreg_model():
    def fn(X_tr, y_tr, X_te):
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        m = LogisticRegression(
            class_weight='balanced', C=0.1, max_iter=1000, random_state=42,
        )
        m.fit(X_tr_s, y_tr.astype(int))
        return m.predict(X_te_s)
    return fn


def ensemble_model(strategy_config):
    def fn(X_tr, y_tr, X_te):
        m = MarketPulseEnsemble.from_strategy_config(strategy_config)
        m.fit(X_tr, y_tr)
        return m.predict(X_te)
    return fn

#  ── Main ──
print("=" * 80)
print("APPROACH COMPARISON")
print("=" * 80)

TICKERS = [
    ("AAPL", "stocks", "short_term"),
    ("MSFT", "stocks", "short_term"),
    ("GOOGL", "stocks", "short_term"),
    ("^GSPC", "indices", "indices_short_term"),
    ("BTC-USD", "crypto", "crypto_short_term"),
]

results_by_approach = {}

for ticker, market, strategy in TICKERS:
    print(f"\n{'─' * 60}")
    print(f"  {ticker}")
    print(f"{'─' * 60}")

    X_all, y, strategy_config = build_data(ticker, market, strategy, horizon=5)
    avail_cols = set(X_all.columns)

    # Filter curated features to those available
    curated_min = [c for c in CURATED_FEATURES_MINIMAL if c in avail_cols]
    curated_ext = [c for c in CURATED_FEATURES_EXTENDED if c in avail_cols]

    approaches = {
        "Curated 12 + XGB d=2":
            (X_all[curated_min], xgb_model(max_depth=2, n_est=100, lr=0.02)),
        "Curated 12 + XGB d=3":
            (X_all[curated_min], xgb_model(max_depth=3, n_est=150, lr=0.03)),
        "Curated 12 + LogReg":
            (X_all[curated_min], logreg_model()),
        "Curated 21 + XGB d=2":
            (X_all[curated_ext], xgb_model(max_depth=2, n_est=100, lr=0.02)),
        "Curated 21 + XGB d=3":
            (X_all[curated_ext], xgb_model(max_depth=3, n_est=150, lr=0.03)),
        "Curated 21 + LogReg":
            (X_all[curated_ext], logreg_model()),
        "All feats + Ensemble (current)":
            (X_all, ensemble_model(strategy_config)),
    }

    # Also try automated FS with fewer features
    init_train = strategy_config.get("validation", {}).get("initial_train_days", 504)
    for nf in [8, 12]:
        n_sel = min(init_train, len(X_all))
        try:
            selected, _ = select_features_pipeline(
                X_all.iloc[:n_sel], y.iloc[:n_sel],
                max_features=nf, corr_threshold=0.85, method='importance',
            )
            approaches[f"Auto FS {nf} + XGB d=3"] = (
                X_all[selected], xgb_model(max_depth=3, n_est=150, lr=0.03)
            )
        except:
            pass

    for name, (X, model_fn) in approaches.items():
        # Drop NaN rows per feature set
        valid = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid]
        y_clean = y[valid]

        r = run_wf_test(X_clean, y_clean, model_fn, purge=5)
        if r:
            acc, base, lift, folds = r
            marker = "✓" if acc > 0.50 else " "
            base_marker = "★" if lift > 0 else " "
            print(f"  {marker}{base_marker} {name:35s} → acc={acc:.1%} base={base:.1%} lift={lift:+.1%} ({folds} folds)")
            if name not in results_by_approach:
                results_by_approach[name] = []
            results_by_approach[name].append(acc)

print("\n\n" + "=" * 80)
print("APPROACH AVERAGES (across tickers)")
print("=" * 80)
print(f"{'Approach':<40} {'Avg Acc':>8} {'Min':>8} {'Max':>8} {'All>50%':>8}")
print("─" * 70)
for name, accs in sorted(results_by_approach.items(), key=lambda x: -np.mean(x[1])):
    avg = np.mean(accs)
    mn = np.min(accs)
    mx = np.max(accs)
    all_above = "YES" if all(a > 0.5 for a in accs) else f"{sum(1 for a in accs if a>0.5)}/{len(accs)}"
    print(f"  {name:<38} {avg:>7.1%} {mn:>7.1%} {mx:>7.1%} {all_above:>8}")
