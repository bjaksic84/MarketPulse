#!/usr/bin/env python3
"""Quick diagnostic: compare no-calibration vs calibration vs XGB-only."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
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
from src.utils.validation import WalkForwardValidator
from src.features.feature_selection import select_features_pipeline


def build_data(ticker, market_name, strategy_name, horizon=5):
    market_config = load_market_config(market_name)
    strategy_config = load_strategy_config(strategy_name)
    years = strategy_config.get("data", {}).get("years_of_history", 5)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
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


def calibrate_on_holdout(model, X_train_full, y_train_full, val_frac=0.15):
    """Train on 85%, calibrate threshold on 15% holdout, return threshold."""
    n = len(X_train_full)
    val_size = max(42, int(n * val_frac))
    train_end = n - val_size

    X_tr = X_train_full.iloc[:train_end]
    y_tr = y_train_full.iloc[:train_end]
    X_val = X_train_full.iloc[train_end:]
    y_val = y_train_full.iloc[train_end:]

    sw = compute_sample_weight("balanced", y_tr)
    temp_model = XGBClassifier(
        max_depth=3, n_estimators=150, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.6, min_child_weight=10,
        gamma=0.5, reg_alpha=0.5, reg_lambda=3.0,
        objective="binary:logistic", eval_metric="logloss",
        use_label_encoder=False, random_state=42
    )
    temp_model.fit(X_tr, y_tr.astype(int), sample_weight=sw)

    proba = temp_model.predict_proba(X_val)[:, 1]
    best_t, best_acc = 0.5, 0.0
    for t in np.arange(0.30, 0.70, 0.02):
        preds = (proba >= t).astype(int)
        acc = accuracy_score(y_val.astype(int), preds)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t, best_acc


def evaluate(ticker, market_name, strategy_name):
    X_all, y, strategy_config = build_data(ticker, market_name, strategy_name)
    fs_cfg = strategy_config.get("feature_engineering", {}).get("feature_selection", {})
    max_features = fs_cfg.get("max_features", 8)
    
    validator = WalkForwardValidator.from_strategy_config(strategy_config)
    folds = validator.split(X_all)

    results = {"no_cal": [], "cal_honest": [], "cal_honest_t": []}
    true_labels = {"no_cal": [], "cal_honest": []}

    for fold in folds:
        X_train = X_all.iloc[fold.train_start:fold.train_end + 1]
        y_train = y.iloc[fold.train_start:fold.train_end + 1]
        X_test = X_all.iloc[fold.test_start:fold.test_end + 1]
        y_test = y.iloc[fold.test_start:fold.test_end + 1]
        if len(X_train) < 100 or len(X_test) < 10:
            continue

        selected, _ = select_features_pipeline(X_train, y_train, max_features=max_features, corr_threshold=0.85, method='importance')
        X_train_sel = X_train[selected]
        X_test_sel = X_test[selected]

        # Train on full training data
        sw = compute_sample_weight("balanced", y_train.astype(int))
        model = XGBClassifier(
            max_depth=3, n_estimators=150, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.6, min_child_weight=10,
            gamma=0.5, reg_alpha=0.5, reg_lambda=3.0,
            objective="binary:logistic", eval_metric="logloss",
            use_label_encoder=False, random_state=42
        )
        model.fit(X_train_sel, y_train.astype(int), sample_weight=sw)

        # Approach 1: no calibration (threshold = 0.5)
        proba = model.predict_proba(X_test_sel)[:, 1]
        preds_nocal = (proba >= 0.5).astype(int)
        results["no_cal"].extend(preds_nocal)
        true_labels["no_cal"].extend(y_test.astype(int).values)

        # Approach 2: honest calibration (train temp model on 85%, calibrate on 15%)
        threshold, _ = calibrate_on_holdout(model, X_train_sel, y_train, val_frac=0.15)
        preds_cal = (proba >= threshold).astype(int)
        results["cal_honest"].extend(preds_cal)
        results["cal_honest_t"].append(threshold)
        true_labels["cal_honest"].extend(y_test.astype(int).values)

    acc_nocal = accuracy_score(true_labels["no_cal"], results["no_cal"])
    acc_cal = accuracy_score(true_labels["cal_honest"], results["cal_honest"])
    avg_t = np.mean(results["cal_honest_t"]) if results["cal_honest_t"] else 0.5

    return acc_nocal, acc_cal, avg_t


tickers = {
    "AAPL": ("stocks", "short_term"),
    "MSFT": ("stocks", "short_term"),
    "GOOGL": ("stocks", "short_term"),
    "^GSPC": ("indices", "indices_short_term"),
    "BTC-USD": ("crypto", "crypto_short_term"),
    "GC=F": ("futures", "futures_short_term"),
}

print(f"{'Ticker':<10} {'No-Cal':>8} {'Honest-Cal':>11} {'Avg-T':>7}")
print("-" * 42)

all_nocal, all_cal = [], []
for ticker, (market, strategy) in tickers.items():
    try:
        acc_nocal, acc_cal, avg_t = evaluate(ticker, market, strategy)
        all_nocal.append(acc_nocal)
        all_cal.append(acc_cal)
        nc = "OK" if acc_nocal > 0.5 else "XX"
        hc = "OK" if acc_cal > 0.5 else "XX"
        print(f"{ticker:<10} {acc_nocal:>7.1%} [{nc}]  {acc_cal:>7.1%} [{hc}]  {avg_t:>5.2f}")
    except Exception as e:
        print(f"{ticker:<10} ERROR: {e}")

print("-" * 42)
print(f"{'Average':<10} {np.mean(all_nocal):>7.1%}      {np.mean(all_cal):>7.1%}")
