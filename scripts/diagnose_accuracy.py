#!/usr/bin/env python3
"""
Diagnostic script: measure walk-forward accuracy across markets.

Runs the same pipeline as the dashboard for representative tickers
and reports accuracy + baseline + feature counts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.data.fetcher import YFinanceFetcher
from src.data.market_config import load_market_config, load_strategy_config
from src.data.preprocessing import preprocess_ohlcv
from src.features.technical import compute_technical_indicators
from src.features.returns import compute_return_features
from src.features.market_adaptive import compute_market_adaptive_features
from src.features.macro import compute_macro_features
from src.features.labels import generate_labels, get_clean_features_and_labels
from src.analysis.regime import detect_regime
from src.models.evaluator import MarketPulseEvaluator
from src.models.xgboost_classifier import MarketPulseXGBClassifier
from src.models.ensemble import MarketPulseEnsemble
from src.utils.validation import WalkForwardValidator


# Representative tickers per market
TEST_CASES = [
    # (market, strategy, ticker, description)
    ("indices", "indices_short_term", "^GSPC", "S&P 500"),
    ("futures", "futures_short_term", "GC=F", "Gold Futures"),
    ("futures", "futures_short_term", "CL=F", "Crude Oil"),
    ("stocks", "short_term", "AAPL", "Apple"),
    ("stocks", "short_term", "MSFT", "Microsoft"),
    ("crypto", "crypto_short_term", "BTC-USD", "Bitcoin"),
]


def run_diagnosis(market_name, strategy_name, ticker, description,
                  use_ensemble=True, use_adaptive=True, use_macro=True):
    """Run walk-forward validation for a single ticker, return metrics dict."""

    market_config = load_market_config(market_name)
    strategy_config = load_strategy_config(strategy_name)

    years = strategy_config.get("data", {}).get("years_of_history", 5)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

    horizon = strategy_config.get("default_horizon", 1)
    num_classes = strategy_config.get("num_classes", 3)
    threshold = strategy_config.get("threshold", 0.01)

    # Fetch
    fetcher = YFinanceFetcher(market_config=market_config)
    raw = fetcher.fetch(ticker, start=start_date, end=end_date)
    if raw.empty:
        return {"ticker": ticker, "error": "No data"}

    # Pipeline
    df = preprocess_ohlcv(raw, market_config=market_config)
    df = compute_technical_indicators(df)
    df = compute_return_features(df)

    n_base_features = len([c for c in df.columns if c not in
                           ("open", "high", "low", "close", "volume", "adj_close", "returns")])

    if use_adaptive:
        df = compute_market_adaptive_features(
            df, market_name=market_name, strategy_config=strategy_config
        )
    n_after_adaptive = len([c for c in df.columns if c not in
                            ("open", "high", "low", "close", "volume", "adj_close", "returns")])

    df = detect_regime(df, market_name=market_name)

    if use_macro and "macro" in strategy_config.get("features", []):
        df = compute_macro_features(df, strategy_config=strategy_config)

    n_total_cols = df.shape[1]

    # Labels
    df = generate_labels(df, horizon=horizon, label_type="classification",
                         num_classes=num_classes, threshold=threshold)
    X, y = get_clean_features_and_labels(df)

    n_features_raw = X.shape[1]
    n_samples = len(X)

    # Feature selection (mirrors trainer + dashboard)
    fs_cfg = strategy_config.get("feature_selection", {})
    fs_method = fs_cfg.get("method", "none")
    if fs_method != "none":
        from src.features.feature_selection import select_features_pipeline
        max_feat = fs_cfg.get("max_features", 15)
        corr_thresh = fs_cfg.get("corr_threshold", 0.90)
        val_init = strategy_config.get("validation", {}).get("initial_train_days", 504)
        n_sel = min(val_init, len(X))
        selected, _ = select_features_pipeline(
            X.iloc[:n_sel], y.iloc[:n_sel],
            max_features=max_feat,
            corr_threshold=corr_thresh,
            method=fs_method,
        )
        X = X[selected]

    n_features = X.shape[1]

    # Class distribution
    class_dist = y.value_counts(normalize=True).sort_index()

    # Walk-forward
    val_cfg = strategy_config.get("validation", {})
    validator = WalkForwardValidator(
        initial_train_days=val_cfg.get("initial_train_days", 504),
        test_days=val_cfg.get("test_days", 21),
        step_days=val_cfg.get("step_days", 21),
    )

    if n_samples < validator.initial_train_days + validator.test_days:
        return {"ticker": ticker, "error": f"Insufficient data: {n_samples}"}

    folds = validator.split(X)
    evaluator = MarketPulseEvaluator(num_classes=num_classes)

    fold_accs = []
    fold_f1s = []
    all_true = []
    all_pred = []

    for fold in folds:
        X_train, y_train, X_test, y_test = validator.get_fold_data(X, y, fold)

        if use_ensemble:
            model = MarketPulseEnsemble.from_strategy_config(strategy_config)
        else:
            model = MarketPulseXGBClassifier.from_strategy_config(strategy_config)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        result = evaluator.evaluate_fold(
            y_true=y_test.values.astype(int),
            y_pred=y_pred,
            y_proba=y_proba,
            fold_number=fold.fold_number,
            train_size=fold.train_size,
        )
        fold_accs.append(result.accuracy)
        fold_f1s.append(result.f1)
        all_true.extend(y_test.values.astype(int))
        all_pred.extend(y_pred)

    baseline = pd.Series(all_true).value_counts(normalize=True).max()

    return {
        "ticker": ticker,
        "description": description,
        "market": market_name,
        "strategy": strategy_name,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_base_features": n_base_features,
        "n_adaptive_added": n_after_adaptive - n_base_features,
        "n_folds": len(folds),
        "mean_accuracy": np.mean(fold_accs),
        "std_accuracy": np.std(fold_accs),
        "mean_f1": np.mean(fold_f1s),
        "baseline": baseline,
        "lift": np.mean(fold_accs) - baseline,
        "class_dist": class_dist.to_dict(),
        "horizon": horizon,
        "threshold": threshold,
        "num_classes": num_classes,
        "ensemble": use_ensemble,
    }


def main():
    print("=" * 80)
    print("MarketPulse Accuracy Diagnostic")
    print("=" * 80)

    results = []
    for market, strategy, ticker, desc in TEST_CASES:
        print(f"\n{'─' * 60}")
        print(f"Testing: {desc} ({ticker}) | Market: {market} | Strategy: {strategy}")
        print(f"{'─' * 60}")

        try:
            r = run_diagnosis(market, strategy, ticker, desc)
            results.append(r)

            if "error" in r:
                print(f"  ERROR: {r['error']}")
            else:
                print(f"  Samples: {r['n_samples']:,}  |  Features: {r['n_features']}")
                thresh_str = "adaptive" if isinstance(r['threshold'], str) else f"{r['threshold']:.1%}"
                print(f"  Folds: {r['n_folds']}  |  Horizon: {r['horizon']}d  |  Threshold: {thresh_str}")
                print(f"  Class dist: {r['class_dist']}")
                print(f"  ──────────────────────────────────────")
                print(f"  ACCURACY: {r['mean_accuracy']:.1%} ± {r['std_accuracy']:.1%}")
                print(f"  F1 Score: {r['mean_f1']:.3f}")
                print(f"  Baseline: {r['baseline']:.1%}")
                print(f"  Lift:     {r['lift']:+.1%}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({"ticker": ticker, "error": str(e)})

    # Summary table
    print("\n\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Ticker':<12} {'Market':<10} {'Acc':>7} {'Base':>7} {'Lift':>7} {'F1':>7} {'Feats':>6} {'Folds':>6} {'Samples':>8}")
    print("─" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['ticker']:<12} {'ERROR':<10} {r['error']}")
        else:
            print(f"{r['ticker']:<12} {r['market']:<10} "
                  f"{r['mean_accuracy']:>6.1%} {r['baseline']:>6.1%} "
                  f"{r['lift']:>+6.1%} {r['mean_f1']:>6.3f} "
                  f"{r['n_features']:>6} {r['n_folds']:>6} {r['n_samples']:>8,}")


if __name__ == "__main__":
    main()
