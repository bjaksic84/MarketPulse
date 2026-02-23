#!/usr/bin/env python3
"""Test sliding window + probability threshold calibration."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.data.fetcher import YFinanceFetcher
from src.data.market_config import load_market_config, load_strategy_config
from src.data.preprocessing import preprocess_ohlcv
from src.features.technical import compute_technical_indicators
from src.features.returns import compute_return_features
from src.features.labels import generate_labels, get_clean_features_and_labels
from src.analysis.regime import detect_regime
from src.features.market_adaptive import compute_market_adaptive_features
from src.utils.validation import WalkForwardValidator, WalkForwardFold
from src.features.feature_selection import select_features_pipeline


def build_data(ticker, market_name, strategy_name, horizon=5):
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
    df = generate_labels(df, horizon=horizon, label_type='classification', num_classes=2, threshold=0.0)
    X_all, y = get_clean_features_and_labels(df)
    return X_all, y, strategy_config


def xgb_with_threshold(X_train, y_train, X_test, threshold=0.5):
    """Train XGB and predict with calibrated threshold."""
    m = XGBClassifier(
        max_depth=3, n_estimators=150, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.6, min_child_weight=10,
        gamma=0.5, reg_alpha=0.5, reg_lambda=3.0,
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    sw = compute_sample_weight("balanced", y_train.astype(int))
    m.fit(X_train, y_train.astype(int), sample_weight=sw, verbose=False)
    proba = m.predict_proba(X_test)[:, 1]  # P(UP)
    return (proba >= threshold).astype(int), proba


def find_optimal_threshold(X_train, y_train):
    """Find the best threshold using the last 20% of training data as validation."""
    n = len(X_train)
    val_size = max(42, int(n * 0.15))
    X_tr = X_train.iloc[:-val_size]
    y_tr = y_train.iloc[:-val_size]
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]

    m = XGBClassifier(
        max_depth=3, n_estimators=150, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.6, min_child_weight=10,
        gamma=0.5, reg_alpha=0.5, reg_lambda=3.0,
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    sw = compute_sample_weight("balanced", y_tr.astype(int))
    m.fit(X_tr, y_tr.astype(int), sample_weight=sw, verbose=False)
    proba = m.predict_proba(X_val)[:, 1]

    best_t, best_acc = 0.5, 0.0
    for t in np.arange(0.35, 0.65, 0.02):
        preds = (proba >= t).astype(int)
        acc = accuracy_score(y_val.astype(int), preds)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t


def run_test(ticker, market, strategy, desc,
             sliding=False, calibrate_threshold=False, max_features=8):
    X_all, y, strategy_config = build_data(ticker, market, strategy, horizon=5)

    # Feature selection
    init_train = strategy_config.get("validation", {}).get("initial_train_days", 504)
    n_sel = min(init_train, len(X_all))
    try:
        selected, _ = select_features_pipeline(
            X_all.iloc[:n_sel], y.iloc[:n_sel],
            max_features=max_features, corr_threshold=0.85, method='importance',
        )
        X = X_all[selected]
    except:
        X = X_all

    valid = X.notna().all(axis=1) & y.notna()
    X, y = X[valid], y[valid]

    validator = WalkForwardValidator(
        initial_train_days=init_train,
        test_days=42,
        step_days=21,
        expanding=not sliding,
        purge_days=5,
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

        if calibrate_threshold:
            threshold = find_optimal_threshold(X_train, y_train)
        else:
            threshold = 0.5

        y_pred, proba = xgb_with_threshold(X_train, y_train, X_test, threshold)
        acc = accuracy_score(y_test.values.astype(int), y_pred)
        fold_accs.append(acc)
        all_true.extend(y_test.values.astype(int))
        all_pred.extend(y_pred)

    mean_acc = np.mean(fold_accs)
    baseline = pd.Series(all_true).value_counts(normalize=True).max()
    return mean_acc, baseline, mean_acc - baseline, len(folds)


TICKERS = [
    ("AAPL", "stocks", "short_term", "Apple"),
    ("MSFT", "stocks", "short_term", "Microsoft"),
    ("GOOGL", "stocks", "short_term", "Google"),
    ("^GSPC", "indices", "indices_short_term", "S&P 500"),
    ("BTC-USD", "crypto", "crypto_short_term", "Bitcoin"),
    ("GC=F", "futures", "futures_short_term", "Gold"),
]

configs = [
    ("Expanding + FS8 + t=0.5",   False, False, 8),
    ("Sliding + FS8 + t=0.5",     True,  False, 8),
    ("Expanding + FS8 + calibrate", False, True, 8),
    ("Sliding + FS8 + calibrate",   True,  True, 8),
    ("Expanding + FS10 + t=0.5",  False, False, 10),
    ("Sliding + FS10 + calibrate",  True,  True, 10),
]

print("=" * 90)
print("SLIDING WINDOW + THRESHOLD CALIBRATION TEST")
print("=" * 90)

results_by_config = {c[0]: [] for c in configs}

for ticker, market, strategy, desc in TICKERS:
    print(f"\n{'─' * 60}")
    print(f"  {desc} ({ticker})")
    print(f"{'─' * 60}")

    for name, sliding, calibrate, max_f in configs:
        try:
            r = run_test(ticker, market, strategy, desc, sliding, calibrate, max_f)
            if r:
                acc, base, lift, folds = r
                marker = "✓" if acc > 0.50 else " "
                print(f"  {marker} {name:<35s} → acc={acc:.1%} lift={lift:+.1%}")
                results_by_config[name].append(acc)
        except Exception as e:
            print(f"  x {name:<35s} → FAILED: {e}")

print("\n\n" + "=" * 90)
print("FINAL AVERAGES")
print("=" * 90)
print(f"{'Config':<40} {'Avg':>7} {'Min':>7} {'Max':>7} {'>50%':>8}")
print("─" * 70)
for name, accs in sorted(results_by_config.items(), key=lambda x: -np.mean(x[1]) if x[1] else 0):
    if accs:
        avg = np.mean(accs)
        print(f"  {name:<38} {avg:>6.1%} {min(accs):>6.1%} {max(accs):>6.1%} {sum(1 for a in accs if a>0.5)}/{len(accs)}")
