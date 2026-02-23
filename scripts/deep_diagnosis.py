#!/usr/bin/env python3
"""Deep diagnostic: identify root causes of low accuracy."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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


def build_features(ticker, market_name, strategy_name):
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
    return df, strategy_config


def test_scenario(name, X, y, strategy_config, use_ensemble=True, use_fs=True):
    val_cfg = strategy_config.get('validation', {})
    init_train = val_cfg.get('initial_train_days', 504)

    if use_fs:
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

    validator = WalkForwardValidator(
        initial_train_days=init_train,
        test_days=val_cfg.get('test_days', 21),
        step_days=val_cfg.get('step_days', 21),
    )

    if len(X) < init_train + val_cfg.get('test_days', 21):
        return None

    folds = validator.split(X)
    all_true, all_pred = [], []
    fold_accs = []

    for fold in folds:
        X_train = X.iloc[fold.train_start:fold.train_end+1]
        y_train = y.iloc[fold.train_start:fold.train_end+1]
        X_test = X.iloc[fold.test_start:fold.test_end+1]
        y_test = y.iloc[fold.test_start:fold.test_end+1]

        if use_ensemble:
            model = MarketPulseEnsemble.from_strategy_config(strategy_config)
        else:
            model = MarketPulseXGBClassifier.from_strategy_config(strategy_config)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test.values.astype(int), y_pred)
        fold_accs.append(acc)
        all_true.extend(y_test.values.astype(int))
        all_pred.extend(y_pred)

    mean_acc = np.mean(fold_accs)
    baseline = pd.Series(all_true).value_counts(normalize=True).max()
    print(f"  [{name}] Acc={mean_acc:.3f} ± {np.std(fold_accs):.3f} | "
          f"Base={baseline:.3f} | Lift={mean_acc-baseline:+.3f} | "
          f"Features={X.shape[1]} | Folds={len(folds)}")
    return {"name": name, "acc": mean_acc, "baseline": baseline, "features": X.shape[1]}


# ── Test 1: Current setup (3-class, adaptive, horizon=3) ──
print("=" * 70)
print("TEST 1: Current setup (AAPL, 3-class adaptive, horizon=3)")
print("=" * 70)

df, strategy_config = build_features("AAPL", "stocks", "short_term")
df_labeled = generate_labels(df.copy(), horizon=3, label_type='classification', num_classes=3, threshold='adaptive')
X, y = get_clean_features_and_labels(df_labeled)
print(f"Samples: {len(X)}, Features: {X.shape[1]}")
print(f"Label dist: {y.value_counts(normalize=True).sort_index().to_dict()}")
test_scenario("3-class adaptive h=3 ensemble+FS", X.copy(), y.copy(), strategy_config, use_ensemble=True, use_fs=True)
test_scenario("3-class adaptive h=3 XGB+FS", X.copy(), y.copy(), strategy_config, use_ensemble=False, use_fs=True)
test_scenario("3-class adaptive h=3 ensemble no FS", X.copy(), y.copy(), strategy_config, use_ensemble=True, use_fs=False)

# ── Test 2: Binary classification (simpler problem) ──
print("\n" + "=" * 70)
print("TEST 2: Binary (2-class, horizon=3)")
print("=" * 70)
df_labeled2 = generate_labels(df.copy(), horizon=3, label_type='classification', num_classes=2)
X2, y2 = get_clean_features_and_labels(df_labeled2)
print(f"Label dist: {y2.value_counts(normalize=True).sort_index().to_dict()}")

# Modify strategy config for binary
sc2 = dict(strategy_config)
sc2["num_classes"] = 2
test_scenario("2-class h=3 ensemble+FS", X2.copy(), y2.copy(), sc2, use_ensemble=True, use_fs=True)
test_scenario("2-class h=3 ensemble no FS", X2.copy(), y2.copy(), sc2, use_ensemble=True, use_fs=False)

# ── Test 3: Different horizons ──
print("\n" + "=" * 70)
print("TEST 3: Different horizons (2-class)")
print("=" * 70)
for h in [1, 3, 5, 10]:
    df_h = generate_labels(df.copy(), horizon=h, label_type='classification', num_classes=2)
    Xh, yh = get_clean_features_and_labels(df_h)
    test_scenario(f"2-class h={h} ensemble+FS", Xh.copy(), yh.copy(), sc2, use_ensemble=True, use_fs=True)

# ── Test 4: Fixed threshold vs adaptive (3-class) ──
print("\n" + "=" * 70)
print("TEST 4: Fixed threshold vs adaptive (3-class, h=3)")
print("=" * 70)
for thresh in ['adaptive', 0.005, 0.01, 0.02, 0.03]:
    df_t = generate_labels(df.copy(), horizon=3, label_type='classification', num_classes=3, threshold=thresh)
    Xt, yt = get_clean_features_and_labels(df_t)
    dist = yt.value_counts(normalize=True).sort_index()
    label = f"thresh={thresh}"
    print(f"  {label}: dist={dist.to_dict()}")
    test_scenario(f"3-class {label}", Xt.copy(), yt.copy(), strategy_config, use_ensemble=True, use_fs=True)

# ── Test 5: Check for data leakage ──
print("\n" + "=" * 70)
print("TEST 5: Data leakage check")
print("=" * 70)
df_check = generate_labels(df.copy(), horizon=3, label_type='classification', num_classes=3, threshold='adaptive')
X_check, y_check = get_clean_features_and_labels(df_check)
# Check if any feature correlates too highly with label
correlations = X_check.corrwith(y_check).abs().sort_values(ascending=False)
print("Top features by correlation with label:")
print(correlations.head(20).to_string())
print(f"\nAny suspicious (>0.5) correlation: {(correlations > 0.5).any()}")
# Check for forward-looking columns in features
fwd_cols = [c for c in X_check.columns if 'fwd' in c.lower() or 'label' in c.lower()]
print(f"Forward-looking columns in features: {fwd_cols}")

# ── Test 6: Feature quality analysis ── 
print("\n" + "=" * 70)
print("TEST 6: NaN analysis in features")
print("=" * 70)
nan_pcts = df.isnull().mean().sort_values(ascending=False)
print(f"Columns with >20% NaN (before label generation):")
print(nan_pcts[nan_pcts > 0.2].to_string())
print(f"\nTotal columns: {len(df.columns)}")
print(f"Rows before dropna: {len(df)}")
X_full, y_full = get_clean_features_and_labels(df_labeled)
print(f"Rows after dropna: {len(X_full)}")
print(f"Data loss from NaN: {1 - len(X_full)/len(df):.1%}")
