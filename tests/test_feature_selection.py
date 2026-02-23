"""
Tests for feature selection and hyperparameter tuning modules.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_selection import (
    filter_correlated,
    select_by_importance,
    select_by_mutual_info,
    select_features_pipeline,
)


# ─────────────── Fixtures ───────────────

@pytest.fixture
def sample_data():
    """Create a synthetic dataset with known properties."""
    np.random.seed(42)
    n = 500

    # Features: some informative, some noise, some correlated
    X = pd.DataFrame({
        "informative_1": np.random.randn(n),
        "informative_2": np.random.randn(n),
        "noise_1": np.random.randn(n),
        "noise_2": np.random.randn(n),
        "noise_3": np.random.randn(n),
    })

    # Create correlated features
    X["corr_with_1"] = X["informative_1"] * 0.95 + np.random.randn(n) * 0.05
    X["corr_with_2"] = X["informative_2"] * 0.90 + np.random.randn(n) * 0.10

    # Target depends on informative features
    score = X["informative_1"] * 2 + X["informative_2"] * 1.5 + np.random.randn(n) * 0.5
    y = pd.Series(np.where(score > 0.5, 2, np.where(score < -0.5, 0, 1)), name="label")

    return X, y


# ─────────────── Test filter_correlated ───────────────

class TestFilterCorrelated:
    def test_removes_correlated_features(self, sample_data):
        X, y = sample_data
        surviving = filter_correlated(X, threshold=0.85)
        # At least one of the correlated pair should be removed
        assert len(surviving) < len(X.columns)

    def test_threshold_1_keeps_all(self, sample_data):
        X, y = sample_data
        surviving = filter_correlated(X, threshold=1.0)
        assert len(surviving) == len(X.columns)

    def test_keeps_more_important_feature(self, sample_data):
        X, y = sample_data
        # informative_1 should be kept over corr_with_1
        importance_order = ["informative_1", "informative_2", "corr_with_1", "corr_with_2",
                            "noise_1", "noise_2", "noise_3"]
        surviving = filter_correlated(
            X, threshold=0.85, importance_order=importance_order
        )
        assert "informative_1" in surviving

    def test_returns_list(self, sample_data):
        X, y = sample_data
        result = filter_correlated(X, threshold=0.9)
        assert isinstance(result, list)
        assert all(isinstance(f, str) for f in result)


# ─────────────── Test select_by_importance ───────────────

class TestSelectByImportance:
    def test_returns_correct_count(self, sample_data):
        X, y = sample_data
        selected = select_by_importance(X, y, max_features=3)
        assert len(selected) == 3

    def test_returns_subset_of_columns(self, sample_data):
        X, y = sample_data
        selected = select_by_importance(X, y, max_features=5)
        assert all(f in X.columns for f in selected)

    def test_returns_list_of_strings(self, sample_data):
        X, y = sample_data
        selected = select_by_importance(X, y, max_features=3)
        assert isinstance(selected, list)
        assert all(isinstance(f, str) for f in selected)


# ─────────────── Test select_by_mutual_info ───────────────

class TestSelectByMutualInfo:
    def test_returns_correct_count(self, sample_data):
        X, y = sample_data
        selected = select_by_mutual_info(X, y, max_features=3)
        assert len(selected) == 3

    def test_returns_subset_of_columns(self, sample_data):
        X, y = sample_data
        selected = select_by_mutual_info(X, y, max_features=4)
        assert all(f in X.columns for f in selected)

    def test_informative_ranked_higher(self, sample_data):
        X, y = sample_data
        selected = select_by_mutual_info(X, y, max_features=3)
        # At least one of the informative features should be selected
        informative = {"informative_1", "informative_2", "corr_with_1", "corr_with_2"}
        assert len(set(selected) & informative) > 0


# ─────────────── Test select_features_pipeline ───────────────

class TestSelectFeaturesPipeline:
    def test_returns_tuple(self, sample_data):
        X, y = sample_data
        result = select_features_pipeline(X, y, max_features=3)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_selected_features_list(self, sample_data):
        X, y = sample_data
        selected, scores = select_features_pipeline(X, y, max_features=3)
        assert isinstance(selected, list)
        assert len(selected) <= 3

    def test_scores_dataframe(self, sample_data):
        X, y = sample_data
        selected, scores = select_features_pipeline(X, y, max_features=5)
        assert isinstance(scores, pd.DataFrame)
        assert "feature" in scores.columns
        assert "xgb_importance" in scores.columns
        assert "mutual_info" in scores.columns
        assert "selected" in scores.columns

    def test_importance_method(self, sample_data):
        X, y = sample_data
        selected, _ = select_features_pipeline(
            X, y, max_features=3, method="importance"
        )
        assert len(selected) > 0

    def test_mutual_info_method(self, sample_data):
        X, y = sample_data
        selected, _ = select_features_pipeline(
            X, y, max_features=3, method="mutual_info"
        )
        assert len(selected) > 0

    def test_invalid_method_raises(self, sample_data):
        X, y = sample_data
        with pytest.raises(ValueError):
            select_features_pipeline(X, y, max_features=3, method="invalid")


# ─────────────── Test Tuner (lightweight) ───────────────

class TestTuner:
    def test_random_search(self, sample_data):
        """Test random search with minimal configuration."""
        from src.models.tuner import MarketPulseTuner

        X, y = sample_data
        strategy_config = {
            "num_classes": 3,
            "validation": {
                "method": "walk_forward",
                "initial_train_days": 200,
                "test_days": 50,
                "step_days": 50,
                "min_train_samples": 100,
            },
        }

        tuner = MarketPulseTuner(
            X=X,
            y=y,
            strategy_config=strategy_config,
            search_space={
                "max_depth": {"type": "int", "low": 3, "high": 6},
                "n_estimators": {"type": "int", "low": 50, "high": 100, "step": 50},
                "learning_rate": {"type": "float", "low": 0.05, "high": 0.2},
            },
            metric="f1_macro",
            max_folds=2,
        )

        best_params = tuner.tune_random(n_trials=3)

        assert isinstance(best_params, dict)
        assert "max_depth" in best_params
        assert tuner.best_score > 0

    def test_results_dataframe(self, sample_data):
        """Verify get_results_df after tuning."""
        from src.models.tuner import MarketPulseTuner

        X, y = sample_data
        strategy_config = {
            "num_classes": 3,
            "validation": {
                "method": "walk_forward",
                "initial_train_days": 200,
                "test_days": 50,
                "step_days": 50,
                "min_train_samples": 100,
            },
        }

        tuner = MarketPulseTuner(
            X=X, y=y, strategy_config=strategy_config,
            search_space={
                "max_depth": {"type": "int", "low": 3, "high": 5},
            },
            max_folds=2,
        )
        tuner.tune_random(n_trials=2)

        df = tuner.get_results_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "score" in df.columns
