#!/usr/bin/env python3
"""Validate accuracy improvements with all fixes applied."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, classification_report

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


def run_test(ticker, market_name, strategy_name, description):
    print(f"\n{'─' * 60}")
    print(f"  {description} ({ticker}) | {market_name}/{strategy_name}")
    print(f"{'─' * 60}")

    market_config = load_market_config(market_name)
    strategy_config = load_strategy_config(strategy_name)
    years = strategy_config.get("data", {}).get("years_of_history", 5)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
    horizon = strategy_config.get("default_horizon", 1)
    num_classes = strategy_config.get("num_classes", 2)
    threshold = strategy_config.get("threshold", 0.0)

    fetcher = YFinanceFetcher(market_config=market_config)
    raw = fetcher.fetch(ticker, start=start_date, end=end_date)
    if raw.empty:
        print("  ERROR: No data")
        return None

    df = preprocess_ohlcv(raw, market_config=market_config)
    df = compute_technical_indicators(df)
    df = compute_return_features(df)
    df = compute_market_adaptive_features(df, market_name=market_name, strategy_config=strategy_config)
    df = detect_regime(df, market_name=market_name)

    df = generate_labels(df, horizon=horizon, label_type='classification',
                         num_classes=num_classes, threshold=threshold)
    X, y = get_clean_features_and_labels(df)

    print(f"  Features: {X.shape[1]} | Samples: {len(X)}")

    # Feature selection
    val_cfg = strategy_config.get('validation', {})
    init_train = val_cfg.get('initial_train_days', 504)
    fs_cfg = strategy_config.get("feature_selection", {})
    fs_method = fs_cfg.get("method", "none")
    if fs_method != "none":
        n_sel = min(init_train, len(X))
        selected, _ = select_features_pipeline(
            X.iloc[:n_sel], y.iloc[:n_sel],
            max_features=fs_cfg.get("max_features", 15),
            corr_threshold=fs_cfg.get("corr_threshold", 0.90),
            method=fs_method,
        )
        X = X[selected]
        print(f"  After FS: {X.shape[1]} features")

    # Use purge_days = horizon to prevent label leakage
    validator = WalkForwardValidator(
        initial_train_days=init_train,
        test_days=val_cfg.get('test_days', 42),
        step_days=val_cfg.get('step_days', 21),
        purge_days=horizon,  # Critical: prevent label leakage
    )

    if len(X) < init_train + val_cfg.get('test_days', 42) + horizon:
        print(f"  ERROR: Insufficient data ({len(X)} samples)")
        return None

    folds = validator.split(X)
    all_true, all_pred = [], []
    fold_accs = []

    for fold in folds:
        X_train = X.iloc[fold.train_start:fold.train_end+1]
        y_train = y.iloc[fold.train_start:fold.train_end+1]
        X_test = X.iloc[fold.test_start:fold.test_end+1]
        y_test = y.iloc[fold.test_start:fold.test_end+1]

        model = MarketPulseEnsemble.from_strategy_config(strategy_config)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test.values.astype(int), y_pred)
        fold_accs.append(acc)
        all_true.extend(y_test.values.astype(int))
        all_pred.extend(y_pred)

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    baseline = pd.Series(all_true).value_counts(normalize=True).max()
    lift = mean_acc - baseline

    print(f"  ACCURACY: {mean_acc:.1%} ± {std_acc:.1%}")
    print(f"  Baseline: {baseline:.1%}  |  Lift: {lift:+.1%}")
    print(f"  Folds: {len(folds)}")

    return {
        "ticker": ticker, "market": market_name, "acc": mean_acc,
        "baseline": baseline, "lift": lift, "folds": len(folds),
        "features": X.shape[1], "description": description,
    }


# Run tests
print("=" * 70)
print("POST-FIX ACCURACY VALIDATION (v2: with purge gap)")
print("=" * 70)

TEST_CASES = [
    ("AAPL",    "stocks",  "short_term",           "Apple"),
    ("MSFT",    "stocks",  "short_term",           "Microsoft"),
    ("GOOGL",   "stocks",  "short_term",           "Google"),
    ("AAPL",    "stocks",  "medium_term",          "Apple Medium-Term"),
    ("^GSPC",   "indices", "indices_short_term",   "S&P 500"),
    ("GC=F",    "futures", "futures_short_term",   "Gold Futures"),
    ("BTC-USD", "crypto",  "crypto_short_term",    "Bitcoin"),
]

results = []
for ticker, market, strategy, desc in TEST_CASES:
    try:
        r = run_test(ticker, market, strategy, desc)
        if r:
            results.append(r)
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()

# Summary
print("\n\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Ticker':<12} {'Desc':<18} {'Acc':>7} {'Base':>7} {'Lift':>7} {'Folds':>6}")
print("─" * 65)
for r in results:
    print(f"{r['ticker']:<12} {r['description']:<18} "
          f"{r['acc']:>6.1%} {r['baseline']:>6.1%} "
          f"{r['lift']:>+6.1%} {r['folds']:>6}")

if results:
    avg_acc = np.mean([r['acc'] for r in results])
    avg_base = np.mean([r['baseline'] for r in results])
    avg_lift = np.mean([r['lift'] for r in results])
    above_50 = sum(1 for r in results if r['acc'] > 0.50)
    print("─" * 65)
    print(f"{'AVERAGE':<30} {avg_acc:>6.1%} {avg_base:>6.1%} {avg_lift:>+6.1%}")
    print(f"\nTickers above 50%: {above_50}/{len(results)}")
    print(f"Tickers above baseline: {sum(1 for r in results if r['lift'] > 0)}/{len(results)}")
