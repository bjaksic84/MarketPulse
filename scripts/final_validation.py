#!/usr/bin/env python3
"""Final production validation — verifies the full pipeline achieves >50% accuracy.

This script exercises the real production code (ensemble, trainer, feature selection)
rather than standalone XGB — ensuring all changes are correctly wired together.
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score

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
from src.models.ensemble import MarketPulseEnsemble


def build_data(ticker, market_name, strategy_name, horizon=5):
    """Build features + labels exactly as the production pipeline does."""
    from src.features.macro import compute_macro_features

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
    df = compute_market_adaptive_features(
        df, market_name=market_name, strategy_config=strategy_config
    )
    df = detect_regime(df, market_name=market_name)

    # Add macro features if enabled in strategy
    strategy_features = strategy_config.get("features", [])
    if "macro" in strategy_features:
        df = compute_macro_features(df, strategy_config=strategy_config)

    df = generate_labels(
        df, horizon=horizon, label_type="classification", num_classes=2, threshold=0.0
    )
    X_all, y = get_clean_features_and_labels(df)
    return X_all, y, strategy_config


def evaluate_ticker(ticker, market_name, strategy_name):
    """Run walk-forward validation using the production ensemble."""
    X_all, y, strategy_config = build_data(ticker, market_name, strategy_name)

    fs_cfg = strategy_config.get("feature_engineering", {}).get("feature_selection", {})
    max_features = fs_cfg.get("max_features", 8)
    corr_threshold = fs_cfg.get("corr_threshold", 0.85)
    method = fs_cfg.get("method", "importance")

    validator = WalkForwardValidator.from_strategy_config(strategy_config)
    folds = validator.split(X_all)

    all_preds, all_true = [], []

    for fold_i, fold in enumerate(folds):
        X_train = X_all.iloc[fold.train_start : fold.train_end + 1]
        y_train = y.iloc[fold.train_start : fold.train_end + 1]
        X_test = X_all.iloc[fold.test_start : fold.test_end + 1]
        y_test = y.iloc[fold.test_start : fold.test_end + 1]

        if len(X_train) < 100 or len(X_test) < 10:
            continue

        # Feature selection on training data
        selected, _ = select_features_pipeline(
            X_train,
            y_train,
            max_features=max_features,
            corr_threshold=corr_threshold,
            method=method,
        )

        X_train_sel = X_train[selected]
        X_test_sel = X_test[selected]

        # Create & train ensemble (with threshold calibration)
        ensemble = MarketPulseEnsemble.from_strategy_config(strategy_config)
        ensemble.fit(X_train_sel, y_train, balance_classes=True)

        preds = ensemble.predict(X_test_sel)
        all_preds.extend(preds)
        all_true.extend(y_test.values)

    if not all_preds:
        return None, 0

    acc = accuracy_score(all_true, all_preds)
    return acc, len(all_preds)


def main():
    # ── Indices (primary focus) ──
    tickers = {
        "^GSPC":  ("indices", "indices_short_term"),
        "^DJI":   ("indices", "indices_short_term"),
        "^IXIC":  ("indices", "indices_short_term"),
        "^RUT":   ("indices", "indices_short_term"),
        # ── Stocks (secondary focus) ──
        "AAPL":   ("stocks", "short_term"),
        "MSFT":   ("stocks", "short_term"),
        "GOOGL":  ("stocks", "short_term"),
        "AMZN":   ("stocks", "short_term"),
        "TSLA":   ("stocks", "short_term"),
        "NVDA":   ("stocks", "short_term"),
        "META":   ("stocks", "short_term"),
        "JPM":    ("stocks", "short_term"),
        "GS":     ("stocks", "short_term"),
        "JNJ":    ("stocks", "short_term"),
        "XOM":    ("stocks", "short_term"),
        "V":      ("stocks", "short_term"),
        "MA":     ("stocks", "short_term"),
        "SPY":    ("stocks", "short_term"),
    }

    print("=" * 64)
    print("    FINAL PRODUCTION VALIDATION — FULL PIPELINE")
    print("=" * 64)

    results = {}
    for ticker, (market, strategy) in tickers.items():
        print(f"\n--- {ticker} ({market} / {strategy}) ---")
        try:
            acc, n = evaluate_ticker(ticker, market, strategy)
            if acc is not None:
                results[ticker] = acc
                status = "PASS" if acc > 0.50 else "FAIL"
                print(f"  Accuracy: {acc:.1%}  (n={n})  [{status}]")
            else:
                print("  SKIPPED: insufficient data")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    if results:
        # Split by market
        idx_results = {t: a for t, a in results.items() if t.startswith("^")}
        stk_results = {t: a for t, a in results.items() if not t.startswith("^")}

        for label, subset in [("INDICES", idx_results), ("STOCKS", stk_results)]:
            if not subset:
                continue
            avg = np.mean(list(subset.values()))
            above = sum(1 for v in subset.values() if v > 0.5)
            print(f"\n  {label}: avg={avg:.1%}  {above}/{len(subset)} > 50%")
            for t, a in subset.items():
                tag = "OK" if a > 0.5 else "XX"
                print(f"    [{tag}] {t:10s}  {a:.1%}")

        avg = np.mean(list(results.values()))
        above = sum(1 for v in results.values() if v > 0.5)
        print(f"\n  OVERALL: avg={avg:.1%}  {above}/{len(results)} > 50%")

        if avg > 0.50 and above > len(results) / 2:
            print("\n>>> GOAL ACHIEVED: Average > 50% <<<")
        else:
            print("\n>>> NEEDS MORE WORK <<<")
    else:
        print("No results!")


if __name__ == "__main__":
    main()
