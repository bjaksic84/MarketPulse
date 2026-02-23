"""
Tests for Phase 4 modules:
- XGBoost regressor
- LightGBM regressor
- Regression evaluator
- Macro / calendar features
- Ensemble regression mode
- Updated dashboard imports
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


# ──────────────────── Fixtures ────────────────────


@pytest.fixture
def sample_ohlcv():
    """Generate 1200 days of realistic OHLCV data."""
    np.random.seed(42)
    n = 1200
    dates = pd.bdate_range(end=datetime.now(), periods=n)

    returns = np.random.normal(0.0005, 0.015, n)
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.normal(0, 0.003, n)),
            "high": prices * (1 + abs(np.random.normal(0, 0.008, n))),
            "low": prices * (1 - abs(np.random.normal(0, 0.008, n))),
            "close": prices,
            "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
        },
        index=dates,
    )
    df.index.name = "Date"

    # Add returns column (needed by many features)
    df["returns"] = df["close"].pct_change()
    df["adj_close"] = df["close"]
    df.dropna(inplace=True)
    return df


@pytest.fixture
def enriched_df(sample_ohlcv):
    """Full pipeline: technical + returns + adaptive + macro + regime."""
    from src.features.technical import compute_technical_indicators
    from src.features.returns import compute_return_features
    from src.features.market_adaptive import compute_market_adaptive_features
    from src.features.macro import compute_macro_features
    from src.analysis.regime import detect_regime
    from src.data.market_config import load_strategy_config

    df = compute_technical_indicators(sample_ohlcv)
    df = compute_return_features(df)
    df = compute_market_adaptive_features(
        df, market_name="stocks", strategy_config=load_strategy_config("short_term")
    )
    df = detect_regime(df, market_name="stocks")
    df = compute_macro_features(df, strategy_config=load_strategy_config("medium_term"))
    return df


@pytest.fixture
def regression_xy(enriched_df):
    """Clean features + regression labels (5-day forward return)."""
    from src.features.labels import generate_labels, get_clean_features_and_labels

    labeled = generate_labels(enriched_df, horizon=5, label_type="regression")
    X, y = get_clean_features_and_labels(labeled)
    return X, y


@pytest.fixture
def classification_xy(enriched_df):
    """Clean features + classification labels (5-day, 2-class)."""
    from src.features.labels import generate_labels, get_clean_features_and_labels

    labeled = generate_labels(
        enriched_df,
        horizon=5,
        label_type="classification",
        num_classes=2,
        threshold=0.0,
    )
    X, y = get_clean_features_and_labels(labeled)
    return X, y


# ──────────────────── Macro Features ────────────────────


class TestMacroFeatures:
    """Tests for src.features.macro."""

    def test_compute_calendar_features(self, sample_ohlcv):
        from src.features.macro import compute_calendar_features

        df = compute_calendar_features(sample_ohlcv.copy())
        assert "cal_day_of_week" in df.columns
        assert "cal_month_sin" in df.columns
        assert "cal_month_cos" in df.columns
        assert "cal_quarter_end" in df.columns
        assert "cal_january" in df.columns
        assert "cal_opex_week" in df.columns
        # Day of week should be 0.0 - 1.0 (normalized)
        assert df["cal_day_of_week"].min() >= 0
        assert df["cal_day_of_week"].max() <= 1.0

    def test_compute_vix_proxy(self, sample_ohlcv):
        from src.features.macro import compute_vix_proxy_features

        df = compute_vix_proxy_features(sample_ohlcv.copy())
        assert "vix_proxy_5d" in df.columns
        assert "vix_proxy_20d" in df.columns
        assert "vix_percentile" in df.columns
        # Vol should be non-negative
        valid = df["vix_proxy_20d"].dropna()
        assert (valid >= 0).all()

    def test_compute_trend_context(self, sample_ohlcv):
        from src.features.macro import compute_trend_context_features

        df = compute_trend_context_features(sample_ohlcv.copy())
        assert "dist_ma_50d" in df.columns
        assert "dist_ma_200d" in df.columns
        assert "golden_cross" in df.columns
        assert "drawdown" in df.columns
        # Golden cross is binary
        valid_gc = df["golden_cross"].dropna()
        assert set(valid_gc.unique()).issubset({0, 1, 0.0, 1.0})

    def test_compute_macro_features_full(self, sample_ohlcv):
        from src.features.macro import compute_macro_features
        from src.data.market_config import load_strategy_config

        strategy = load_strategy_config("medium_term")
        df_in = sample_ohlcv.copy()
        original_cols = set(df_in.columns)
        df = compute_macro_features(df_in, strategy_config=strategy)
        new_cols = set(df.columns) - original_cols
        assert len(new_cols) >= 15  # Calendar + VIX + trend

    def test_get_macro_feature_names(self):
        from src.features.macro import get_macro_feature_names

        names = get_macro_feature_names()
        assert isinstance(names, list)
        assert len(names) >= 20
        assert "cal_day_of_week" in names
        assert "vix_percentile" in names

    def test_macro_features_no_nans_in_output_tail(self, sample_ohlcv):
        """After computing macro features, the last 50 rows should be clean."""
        from src.features.macro import compute_macro_features
        from src.data.market_config import load_strategy_config

        strategy = load_strategy_config("medium_term")
        df = compute_macro_features(sample_ohlcv, strategy_config=strategy)
        from src.features.macro import get_macro_feature_names

        macro_cols = [c for c in get_macro_feature_names() if c in df.columns]
        tail = df[macro_cols].iloc[-50:]
        # Allow some NaN (lookback periods) but most should be clean
        pct_valid = tail.notna().mean().mean()
        assert pct_valid > 0.7, f"Only {pct_valid:.0%} valid in tail"


# ──────────────────── XGBoost Regressor ────────────────────


class TestXGBRegressor:
    """Tests for src.models.xgboost_regressor."""

    def test_fit_predict(self, regression_xy):
        from src.models.xgboost_regressor import MarketPulseXGBRegressor

        X, y = regression_xy
        assert len(X) >= 500, f"Need at least 500 samples, got {len(X)}"
        split = len(X) - 50
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_test = X.iloc[split:split + 20]

        model = MarketPulseXGBRegressor()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        assert preds.dtype == np.float64 or preds.dtype == np.float32

    def test_predict_proba_shape(self, regression_xy):
        from src.models.xgboost_regressor import MarketPulseXGBRegressor

        X, y = regression_xy
        split = len(X) - 50
        model = MarketPulseXGBRegressor()
        model.fit(X.iloc[:split], y.iloc[:split])

        proba = model.predict_proba(X.iloc[split:split + 20])
        assert proba.shape == (len(X.iloc[split:split + 20]), 1)

    def test_feature_importance(self, regression_xy):
        from src.models.xgboost_regressor import MarketPulseXGBRegressor

        X, y = regression_xy
        split = len(X) - 50
        model = MarketPulseXGBRegressor()
        model.fit(X.iloc[:split], y.iloc[:split])

        imp = model.get_feature_importance()
        assert isinstance(imp, pd.Series)
        assert len(imp) == X.shape[1]
        assert (imp >= 0).all()

    def test_from_strategy_config(self):
        from src.models.xgboost_regressor import MarketPulseXGBRegressor
        from src.data.market_config import load_strategy_config

        cfg = load_strategy_config("medium_term_regression")
        model = MarketPulseXGBRegressor.from_strategy_config(cfg)
        assert model is not None
        assert model.hyperparameters["max_depth"] == 5

    def test_shap_values(self, regression_xy):
        from src.models.xgboost_regressor import MarketPulseXGBRegressor

        X, y = regression_xy
        split = len(X) - 50
        model = MarketPulseXGBRegressor()
        model.fit(X.iloc[:split], y.iloc[:split])

        shap_vals = model.get_shap_values(X.iloc[split:split + 5])
        assert shap_vals.shape[0] == 5
        assert shap_vals.shape[1] == X.shape[1]


# ──────────────────── LightGBM Regressor ────────────────────


class TestLGBRegressor:
    """Tests for src.models.lightgbm_regressor."""

    def test_fit_predict(self, regression_xy):
        from src.models.lightgbm_regressor import MarketPulseLGBRegressor

        X, y = regression_xy
        split = len(X) - 50
        model = MarketPulseLGBRegressor()
        model.fit(X.iloc[:split], y.iloc[:split])

        X_test = X.iloc[split:split + 20]
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_predict_proba_shape(self, regression_xy):
        from src.models.lightgbm_regressor import MarketPulseLGBRegressor

        X, y = regression_xy
        split = len(X) - 50
        model = MarketPulseLGBRegressor()
        model.fit(X.iloc[:split], y.iloc[:split])

        X_test = X.iloc[split:split + 20]
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 1)

    def test_feature_importance(self, regression_xy):
        from src.models.lightgbm_regressor import MarketPulseLGBRegressor

        X, y = regression_xy
        split = len(X) - 50
        model = MarketPulseLGBRegressor()
        model.fit(X.iloc[:split], y.iloc[:split])

        imp = model.get_feature_importance()
        assert isinstance(imp, pd.Series)
        assert len(imp) == X.shape[1]

    def test_from_strategy_config(self):
        from src.models.lightgbm_regressor import MarketPulseLGBRegressor
        from src.data.market_config import load_strategy_config

        cfg = load_strategy_config("medium_term_regression")
        model = MarketPulseLGBRegressor.from_strategy_config(cfg)
        assert model is not None


# ──────────────────── Regression Evaluator ────────────────────


class TestRegressionEvaluator:
    """Tests for src.models.regression_evaluator."""

    def test_evaluate_fold(self):
        from src.models.regression_evaluator import RegressionEvaluator

        evaluator = RegressionEvaluator()
        y_true = np.array([0.01, -0.02, 0.03, -0.01, 0.005])
        y_pred = np.array([0.008, -0.015, 0.02, 0.001, 0.003])

        result = evaluator.evaluate_fold(y_true, y_pred, fold_number=1)
        assert result.mae > 0
        assert result.rmse > 0
        assert 0 <= result.directional_accuracy <= 1
        assert result.test_size == 5

    def test_directional_accuracy(self):
        from src.models.regression_evaluator import RegressionEvaluator

        evaluator = RegressionEvaluator()
        # All directions correct
        result = evaluator.evaluate_fold(
            np.array([1, -1, 1, -1]),
            np.array([0.5, -0.5, 0.3, -0.1]),
        )
        assert result.directional_accuracy == 1.0

        # All wrong
        result = evaluator.evaluate_fold(
            np.array([1, -1, 1, -1]),
            np.array([-0.5, 0.5, -0.3, 0.1]),
        )
        assert result.directional_accuracy == 0.0

    def test_information_coefficient(self):
        from src.models.regression_evaluator import RegressionEvaluator

        evaluator = RegressionEvaluator()
        # Perfect rank correlation
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([10, 20, 30, 40, 50])
        result = evaluator.evaluate_fold(y_true, y_pred)
        assert abs(result.ic - 1.0) < 0.01  # Should be ~1.0

    def test_aggregate_results(self):
        from src.models.regression_evaluator import (
            RegressionEvaluator,
            RegressionFoldResult,
        )

        evaluator = RegressionEvaluator()

        # Create two fake fold results
        results = []
        for i in range(3):
            y_true = np.random.normal(0, 0.02, 30)
            y_pred = y_true * 0.5 + np.random.normal(0, 0.01, 30)
            r = evaluator.evaluate_fold(y_true, y_pred, fold_number=i)
            results.append(r)

        report = evaluator.aggregate_results(results, ticker="TEST", horizon=5)
        assert report.ticker == "TEST"
        assert report.horizon == 5
        assert len(report.fold_results) == 3
        assert report.mean_mae > 0
        assert report.mean_directional_accuracy > 0

    def test_print_report(self):
        from src.models.regression_evaluator import RegressionEvaluator

        evaluator = RegressionEvaluator()
        y_true = np.random.normal(0, 0.02, 30)
        y_pred = y_true * 0.3 + np.random.normal(0, 0.01, 30)
        r = evaluator.evaluate_fold(y_true, y_pred)
        report = evaluator.aggregate_results([r], ticker="SPY")
        text = evaluator.print_report(report)
        assert "SPY" in text
        assert "MAE" in text


# ──────────────────── Ensemble Regression Mode ────────────────────


class TestEnsembleRegression:
    """Test that ensemble works in regression mode."""

    def test_ensemble_regression_detect(self):
        from src.models.ensemble import MarketPulseEnsemble
        from src.data.market_config import load_strategy_config

        cfg = load_strategy_config("medium_term_regression")
        ensemble = MarketPulseEnsemble.from_strategy_config(cfg)
        assert ensemble._is_regression is True

    def test_ensemble_classification_detect(self):
        from src.models.ensemble import MarketPulseEnsemble
        from src.data.market_config import load_strategy_config

        cfg = load_strategy_config("short_term")
        ensemble = MarketPulseEnsemble.from_strategy_config(cfg)
        assert ensemble._is_regression is False

    def test_ensemble_regression_fit_predict(self, regression_xy):
        from src.models.ensemble import MarketPulseEnsemble
        from src.data.market_config import load_strategy_config

        X, y = regression_xy
        split = len(X) - 50
        cfg = load_strategy_config("medium_term_regression")
        ensemble = MarketPulseEnsemble.from_strategy_config(cfg)

        ensemble.fit(X.iloc[:split], y.iloc[:split])
        X_test = X.iloc[split:split + 20]
        preds = ensemble.predict(X_test)

        assert len(preds) == len(X_test)
        assert preds.dtype in (np.float64, np.float32)

    def test_ensemble_regression_predict_proba(self, regression_xy):
        from src.models.ensemble import MarketPulseEnsemble
        from src.data.market_config import load_strategy_config

        X, y = regression_xy
        split = len(X) - 50
        cfg = load_strategy_config("medium_term_regression")
        ensemble = MarketPulseEnsemble.from_strategy_config(cfg)
        ensemble.fit(X.iloc[:split], y.iloc[:split])

        X_test = X.iloc[split:split + 20]
        proba = ensemble.predict_proba(X_test)
        assert proba.shape == (len(X_test), 1)

    def test_ensemble_regression_feature_importance(self, regression_xy):
        from src.models.ensemble import MarketPulseEnsemble
        from src.data.market_config import load_strategy_config

        X, y = regression_xy
        split = len(X) - 50
        cfg = load_strategy_config("medium_term_regression")
        ensemble = MarketPulseEnsemble.from_strategy_config(cfg)
        ensemble.fit(X.iloc[:split], y.iloc[:split])

        imp = ensemble.get_feature_importance()
        assert isinstance(imp, pd.Series)
        assert len(imp) == X.shape[1]

    def test_ensemble_classification_still_works(self, classification_xy):
        from src.models.ensemble import MarketPulseEnsemble
        from src.data.market_config import load_strategy_config

        X, y = classification_xy
        split = len(X) - 50
        cfg = load_strategy_config("short_term")
        ensemble = MarketPulseEnsemble.from_strategy_config(cfg)
        ensemble.fit(X.iloc[:split], y.iloc[:split])

        X_test = X.iloc[split:split + 20]
        preds = ensemble.predict(X_test)
        proba = ensemble.predict_proba(X_test)

        assert len(preds) == len(X_test)
        n_classes = len(np.unique(y.iloc[:split]))
        assert proba.shape[1] == n_classes
        assert set(preds).issubset(set(np.unique(y)))


# ──────────────────── Strategy Configs ────────────────────


class TestStrategyConfigs:
    """Test Phase 4 strategy configs load correctly."""

    def test_medium_term_loads(self):
        from src.data.market_config import load_strategy_config

        cfg = load_strategy_config("medium_term")
        assert cfg["name"] == "medium_term"
        assert cfg["label_type"] == "classification"
        assert cfg["threshold"] == 0.0
        assert 5 in cfg["horizon_days"]
        assert 10 in cfg["horizon_days"]
        assert "macro" in cfg["features"]
        assert cfg["ensemble"]["enabled"] is True

    def test_medium_term_regression_loads(self):
        from src.data.market_config import load_strategy_config

        cfg = load_strategy_config("medium_term_regression")
        assert cfg["name"] == "medium_term_regression"
        assert cfg["label_type"] == "regression"
        assert 5 in cfg["horizon_days"]
        assert cfg["ensemble"]["enabled"] is True
        # Check regressor model types
        model_types = [m["type"] for m in cfg["ensemble"]["models"]]
        assert "xgboost_regressor" in model_types
        assert "lightgbm_regressor" in model_types


# ──────────────────── Init Exports ────────────────────


class TestInitExports:
    """Verify __init__.py exposes Phase 4 modules."""

    def test_models_init(self):
        from src.models import (
            MarketPulseXGBRegressor,
            MarketPulseLGBRegressor,
            RegressionEvaluator,
        )

        assert MarketPulseXGBRegressor is not None
        assert MarketPulseLGBRegressor is not None
        assert RegressionEvaluator is not None

    def test_features_init(self):
        from src.features import compute_macro_features, get_macro_feature_names

        assert compute_macro_features is not None
        assert get_macro_feature_names is not None


# ──────────────────── Dashboard Imports ────────────────────


class TestDashboardImports:
    """Test that the dashboard module can be imported without errors."""

    def test_dashboard_imports(self):
        """Verify all imports used by the dashboard exist."""
        from src.features.market_adaptive import compute_market_adaptive_features
        from src.features.macro import compute_macro_features
        from src.analysis.regime import detect_regime, REGIME_LABELS
        from src.models.regression_evaluator import RegressionEvaluator
        from src.models.ensemble import MarketPulseEnsemble

        assert REGIME_LABELS is not None
        assert callable(compute_market_adaptive_features)
        assert callable(compute_macro_features)
        assert callable(detect_regime)
