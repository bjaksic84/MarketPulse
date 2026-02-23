"""
Tests for Phase 3 modules:
- Market-adaptive features
- Market regime detector
- LightGBM classifier
- Ensemble model
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


# ──────────────────── Fixtures ────────────────────


@pytest.fixture
def sample_ohlcv():
    """Generate 300 days of realistic OHLCV data."""
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range(end=datetime.now(), periods=n)

    # Random walk price
    returns = np.random.normal(0.0005, 0.015, n)
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "open": prices * (1 + np.random.normal(0, 0.003, n)),
        "high": prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
        "close": prices,
        "volume": np.random.lognormal(15, 1, n).astype(int),
    }, index=dates)

    df["returns"] = df["close"].pct_change()
    return df


@pytest.fixture
def sample_ohlcv_with_dates():
    """Generate data spanning weekends for date-based tests."""
    np.random.seed(123)
    n = 365
    dates = pd.date_range(end=datetime.now(), periods=n, freq="D")  # include weekends

    returns = np.random.normal(0.001, 0.03, n)
    prices = 40000 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "open": prices * (1 + np.random.normal(0, 0.005, n)),
        "high": prices * (1 + np.abs(np.random.normal(0, 0.015, n))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.015, n))),
        "close": prices,
        "volume": np.random.lognormal(20, 1.5, n).astype(int),
    }, index=dates)

    df["returns"] = df["close"].pct_change()
    return df


@pytest.fixture
def strategy_config():
    """Minimal strategy config with ensemble + market_adaptive."""
    return {
        "name": "test_strategy",
        "num_classes": 3,
        "threshold": 0.01,
        "features": ["technical", "returns", "market_adaptive"],
        "model": {
            "type": "xgboost_classifier",
            "hyperparameters": {
                "max_depth": 4,
                "n_estimators": 50,
                "learning_rate": 0.1,
                "random_state": 42,
                "n_jobs": 1,
            },
        },
        "ensemble": {
            "enabled": True,
            "models": [
                {"type": "xgboost_classifier", "weight": 0.5},
                {"type": "lightgbm_classifier", "weight": 0.5},
            ],
        },
        "market_adaptive": {
            "weekend_effects": False,
            "gap_analysis": True,
            "volatility_regime": True,
            "mean_reversion": False,
            "correlation_features": False,
            "volume_spike_multiplier": 2.5,
        },
    }


# ──────────────────── Market-Adaptive Feature Tests ────────────────────


class TestVolatilityRegime:
    def test_adds_columns(self, sample_ohlcv):
        from src.features.market_adaptive import compute_volatility_regime

        df = compute_volatility_regime(sample_ohlcv.copy())
        assert "vol_regime_ratio" in df.columns
        assert "vol_expansion" in df.columns
        assert "vol_regime_label" in df.columns

    def test_values_reasonable(self, sample_ohlcv):
        from src.features.market_adaptive import compute_volatility_regime

        df = compute_volatility_regime(sample_ohlcv.copy())
        valid = df["vol_regime_ratio"].dropna()
        assert valid.min() > 0  # ratio is always positive
        assert df["vol_expansion"].isin([0, 1]).all()
        assert df["vol_regime_label"].dropna().isin([0, 1, 2]).all()


class TestGapFeatures:
    def test_adds_columns(self, sample_ohlcv):
        from src.features.market_adaptive import compute_gap_features

        df = compute_gap_features(sample_ohlcv.copy())
        expected = {"overnight_gap", "overnight_gap_abs", "gap_up", "gap_down",
                    "gap_fill_ratio", "avg_gap_5d", "gap_volatility_20d"}
        assert expected.issubset(set(df.columns))

    def test_gap_direction_flags(self, sample_ohlcv):
        from src.features.market_adaptive import compute_gap_features

        df = compute_gap_features(sample_ohlcv.copy())
        # gap_up and gap_down should be 0 or 1
        assert df["gap_up"].isin([0, 1]).all()
        assert df["gap_down"].isin([0, 1]).all()


class TestVolumeSpike:
    def test_adds_columns(self, sample_ohlcv):
        from src.features.market_adaptive import compute_volume_spike_features

        df = compute_volume_spike_features(sample_ohlcv.copy())
        assert "volume_zscore" in df.columns
        assert "volume_spike" in df.columns
        assert "volume_trend_5d" in df.columns

    def test_custom_threshold(self, sample_ohlcv):
        from src.features.market_adaptive import compute_volume_spike_features

        # Very low multiplier → more spikes
        df_low = compute_volume_spike_features(sample_ohlcv.copy(), spike_multiplier=1.0)
        df_high = compute_volume_spike_features(sample_ohlcv.copy(), spike_multiplier=5.0)
        assert df_low["volume_spike"].sum() >= df_high["volume_spike"].sum()


class TestIndexFeatures:
    def test_adds_columns(self, sample_ohlcv):
        from src.features.market_adaptive import compute_index_features

        df = compute_index_features(sample_ohlcv.copy())
        assert "reversion_zscore_20d" in df.columns
        assert "dist_52w_high" in df.columns
        assert "realized_vol_10d" in df.columns
        assert "days_since_big_move" in df.columns

    def test_dist_52w_high_negative(self, sample_ohlcv):
        from src.features.market_adaptive import compute_index_features

        df = compute_index_features(sample_ohlcv.copy())
        # distance from 52w high should always be <= 0
        valid = df["dist_52w_high"].dropna()
        assert (valid <= 0.001).all()  # small float tolerance


class TestAdaptiveDispatcher:
    def test_stocks(self, sample_ohlcv, strategy_config):
        from src.features.market_adaptive import compute_market_adaptive_features

        n_before = sample_ohlcv.shape[1]
        df = compute_market_adaptive_features(sample_ohlcv.copy(), "stocks", strategy_config)
        assert df.shape[1] > n_before

    def test_indices(self, sample_ohlcv, strategy_config):
        from src.features.market_adaptive import compute_market_adaptive_features

        df = compute_market_adaptive_features(sample_ohlcv.copy(), "indices", strategy_config)
        assert "reversion_zscore_20d" in df.columns

    def test_feature_name_listing(self):
        from src.features.market_adaptive import get_adaptive_feature_names

        for market in ["stocks", "indices"]:
            names = get_adaptive_feature_names(market)
            assert len(names) > 0
            assert isinstance(names, list)


# ──────────────────── Market Regime Detector Tests ────────────────────


class TestRegimeConfig:
    def test_default(self):
        from src.analysis.regime import RegimeConfig

        cfg = RegimeConfig()
        assert cfg.trend_short_window == 20
        assert cfg.smooth_window == 5

    def test_per_market(self):
        from src.analysis.regime import RegimeConfig

        indices = RegimeConfig.for_market("indices")
        assert indices.smooth_window == 7


class TestRegimeDetector:
    def test_detect_adds_columns(self, sample_ohlcv):
        from src.analysis.regime import detect_regime

        df = detect_regime(sample_ohlcv.copy())
        assert "regime" in df.columns
        assert "regime_label" in df.columns
        assert "regime_score" in df.columns
        assert "trend_direction" in df.columns

    def test_regime_values(self, sample_ohlcv):
        from src.analysis.regime import detect_regime

        df = detect_regime(sample_ohlcv.copy())
        assert df["regime"].dropna().isin([0, 1, 2]).all()
        assert df["regime_label"].dropna().isin(["bearish", "neutral", "bullish"]).all()

    def test_regime_score_bounded(self, sample_ohlcv):
        from src.analysis.regime import detect_regime

        df = detect_regime(sample_ohlcv.copy())
        valid = df["regime_score"].dropna()
        assert valid.min() >= -1.0
        assert valid.max() <= 1.0

    def test_different_markets(self, sample_ohlcv):
        from src.analysis.regime import detect_regime

        for market in ["stocks", "indices"]:
            df = detect_regime(sample_ohlcv.copy(), market_name=market)
            assert "regime" in df.columns

    def test_regime_summary(self, sample_ohlcv):
        from src.analysis.regime import MarketRegimeDetector

        detector = MarketRegimeDetector()
        df = detector.detect(sample_ohlcv.copy())
        summary = detector.get_regime_summary(df)
        assert "current_regime" in summary
        assert "regime_changes" in summary
        assert summary["regime_changes"] >= 0

    def test_regime_transitions(self, sample_ohlcv):
        from src.analysis.regime import detect_regime, get_regime_transitions

        df = detect_regime(sample_ohlcv.copy())
        transitions = get_regime_transitions(df)
        assert isinstance(transitions, pd.DataFrame)
        if len(transitions) > 0:
            assert "prev_regime" in transitions.columns

    def test_no_close_column(self):
        from src.analysis.regime import detect_regime

        df = pd.DataFrame({"price": [1, 2, 3]})
        result = detect_regime(df)
        # Should return df unchanged (no crash)
        assert "regime" not in result.columns


# ──────────────────── LightGBM Classifier Tests ────────────────────


class TestLGBClassifier:
    @pytest.fixture
    def training_data(self):
        np.random.seed(42)
        n = 200
        X = pd.DataFrame(np.random.randn(n, 10), columns=[f"f{i}" for i in range(10)])
        y = pd.Series(np.random.choice([0, 1, 2], n))
        return X, y

    def test_fit_predict(self, training_data):
        from src.models.lightgbm_classifier import MarketPulseLGBClassifier

        X, y = training_data
        model = MarketPulseLGBClassifier(num_classes=3)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1, 2})

    def test_predict_proba(self, training_data):
        from src.models.lightgbm_classifier import MarketPulseLGBClassifier

        X, y = training_data
        model = MarketPulseLGBClassifier(num_classes=3)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_feature_importance(self, training_data):
        from src.models.lightgbm_classifier import MarketPulseLGBClassifier

        X, y = training_data
        model = MarketPulseLGBClassifier(num_classes=3)
        model.fit(X, y)
        importance = model.get_feature_importance()
        assert len(importance) == X.shape[1]
        assert (importance >= 0).all()

    def test_from_strategy_config(self, strategy_config, training_data):
        from src.models.lightgbm_classifier import MarketPulseLGBClassifier

        X, y = training_data
        model = MarketPulseLGBClassifier.from_strategy_config(strategy_config)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_binary_classification(self):
        from src.models.lightgbm_classifier import MarketPulseLGBClassifier

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(np.random.choice([0, 1], 100))

        model = MarketPulseLGBClassifier(num_classes=2)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape[1] == 2


# ──────────────────── Ensemble Tests ────────────────────


class TestEnsemble:
    @pytest.fixture
    def training_data(self):
        np.random.seed(42)
        n = 200
        X = pd.DataFrame(np.random.randn(n, 10), columns=[f"f{i}" for i in range(10)])
        y = pd.Series(np.random.choice([0, 1, 2], n))
        return X, y

    def test_fit_predict(self, training_data, strategy_config):
        from src.models.ensemble import MarketPulseEnsemble

        X, y = training_data
        ensemble = MarketPulseEnsemble.from_strategy_config(strategy_config)
        ensemble.fit(X, y)
        preds = ensemble.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1, 2})

    def test_predict_proba(self, training_data, strategy_config):
        from src.models.ensemble import MarketPulseEnsemble

        X, y = training_data
        ensemble = MarketPulseEnsemble.from_strategy_config(strategy_config)
        ensemble.fit(X, y)
        proba = ensemble.predict_proba(X)
        assert proba.shape == (len(X), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_agreement_score(self, training_data, strategy_config):
        from src.models.ensemble import MarketPulseEnsemble

        X, y = training_data
        ensemble = MarketPulseEnsemble.from_strategy_config(strategy_config)
        ensemble.fit(X, y)
        agreement = ensemble.get_agreement_score(X)
        assert len(agreement) == len(X)
        assert (agreement >= 0).all() and (agreement <= 1).all()

    def test_individual_predictions(self, training_data, strategy_config):
        from src.models.ensemble import MarketPulseEnsemble

        X, y = training_data
        ensemble = MarketPulseEnsemble.from_strategy_config(strategy_config)
        ensemble.fit(X, y)
        indiv = ensemble.get_individual_predictions(X)
        assert len(indiv) == 2  # XGBoost + LightGBM

    def test_feature_importance(self, training_data, strategy_config):
        from src.models.ensemble import MarketPulseEnsemble

        X, y = training_data
        ensemble = MarketPulseEnsemble.from_strategy_config(strategy_config)
        ensemble.fit(X, y)
        importance = ensemble.get_feature_importance()
        assert len(importance) == X.shape[1]

    def test_single_model_fallback(self, training_data):
        from src.models.ensemble import MarketPulseEnsemble

        # Config with ensemble disabled → single model
        config = {
            "num_classes": 3,
            "model": {"type": "xgboost_classifier", "hyperparameters": {
                "max_depth": 3, "n_estimators": 20, "n_jobs": 1,
            }},
            "ensemble": {"enabled": False},
        }
        X, y = training_data
        ensemble = MarketPulseEnsemble.from_strategy_config(config)
        ensemble.fit(X, y)
        preds = ensemble.predict(X)
        assert len(preds) == len(X)
        # Should only have 1 model
        assert len(ensemble.models) == 1
