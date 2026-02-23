"""Tests for walk-forward validation."""

import pytest
import numpy as np
import pandas as pd

from src.utils.validation import WalkForwardValidator, WalkForwardFold


def _make_sample_data(n: int = 1000) -> tuple:
    """Create sample X, y for testing."""
    dates = pd.bdate_range("2020-01-02", periods=n)
    X = pd.DataFrame(
        np.random.randn(n, 5),
        columns=["f1", "f2", "f3", "f4", "f5"],
        index=dates,
    )
    y = pd.Series(np.random.randint(0, 3, n), index=dates, name="label")
    return X, y


class TestWalkForwardValidator:
    """Test walk-forward cross-validation."""

    def test_basic_split(self):
        X, y = _make_sample_data(1000)
        validator = WalkForwardValidator(
            initial_train_days=504,
            test_days=21,
            step_days=21,
        )
        folds = validator.split(X)

        assert len(folds) > 0
        assert all(isinstance(f, WalkForwardFold) for f in folds)

    def test_no_overlap(self):
        """Train and test periods must not overlap."""
        X, y = _make_sample_data(1000)
        validator = WalkForwardValidator(
            initial_train_days=504,
            test_days=21,
            step_days=21,
        )
        folds = validator.split(X)

        for fold in folds:
            assert fold.train_end < fold.test_start, (
                f"Fold {fold.fold_number}: train_end={fold.train_end} "
                f"must be < test_start={fold.test_start}"
            )

    def test_train_precedes_test(self):
        """All training data must come before test data (temporal ordering)."""
        X, y = _make_sample_data(1000)
        validator = WalkForwardValidator(
            initial_train_days=504,
            test_days=21,
            step_days=21,
        )
        folds = validator.split(X)

        for fold in folds:
            assert fold.train_end < fold.test_start

    def test_expanding_window(self):
        """In expanding mode, training start should always be 0."""
        X, y = _make_sample_data(1000)
        validator = WalkForwardValidator(
            initial_train_days=504,
            test_days=21,
            step_days=21,
            expanding=True,
        )
        folds = validator.split(X)

        for fold in folds:
            assert fold.train_start == 0

    def test_training_grows(self):
        """In expanding mode, training size should increase across folds."""
        X, y = _make_sample_data(1000)
        validator = WalkForwardValidator(
            initial_train_days=504,
            test_days=21,
            step_days=21,
            expanding=True,
        )
        folds = validator.split(X)

        train_sizes = [f.train_size for f in folds]
        assert train_sizes == sorted(train_sizes), "Train size should grow monotonically"

    def test_get_fold_data(self):
        X, y = _make_sample_data(1000)
        validator = WalkForwardValidator(
            initial_train_days=504,
            test_days=21,
            step_days=21,
        )
        folds = validator.split(X)
        fold = folds[0]

        X_train, y_train, X_test, y_test = validator.get_fold_data(X, y, fold)

        assert len(X_train) == fold.train_size
        assert len(X_test) == fold.test_size
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_min_train_samples(self):
        """Folds with insufficient training data should be skipped."""
        X, y = _make_sample_data(500)
        validator = WalkForwardValidator(
            initial_train_days=504,
            test_days=21,
            step_days=21,
            min_train_samples=600,  # impossible to satisfy
        )
        folds = validator.split(X)
        # May or may not have folds, depending on data length
        for fold in folds:
            assert fold.train_size >= 600

    def test_sklearn_compatible_splits(self):
        X, y = _make_sample_data(1000)
        validator = WalkForwardValidator(
            initial_train_days=504,
            test_days=21,
            step_days=21,
        )
        splits = validator.get_sklearn_splits(X)

        assert len(splits) > 0
        for train_idx, test_idx in splits:
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0
            # Temporal order
            assert train_idx.max() < test_idx.min()

    def test_from_strategy_config(self):
        config = {
            "validation": {
                "initial_train_days": 252,
                "test_days": 10,
                "step_days": 10,
                "min_train_samples": 200,
            }
        }
        validator = WalkForwardValidator.from_strategy_config(config)
        assert validator.initial_train_days == 252
        assert validator.test_days == 10

    def test_fold_has_dates(self):
        """Folds should include date information when index is DatetimeIndex."""
        X, y = _make_sample_data(1000)
        validator = WalkForwardValidator(
            initial_train_days=504,
            test_days=21,
            step_days=21,
        )
        folds = validator.split(X)

        for fold in folds:
            assert fold.train_start_date is not None
            assert fold.test_start_date is not None
            assert fold.train_end_date < fold.test_start_date
