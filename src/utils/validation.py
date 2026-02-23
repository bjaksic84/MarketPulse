"""
Walk-forward validation for time series.

Implements expanding-window and sliding-window walk-forward splits.
CRITICAL: Never shuffles data. Always respects temporal ordering.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """A single train/test fold in walk-forward validation."""

    fold_number: int
    train_start: int    # index position (iloc)
    train_end: int      # inclusive
    test_start: int
    test_end: int       # inclusive
    train_start_date: pd.Timestamp = None
    train_end_date: pd.Timestamp = None
    test_start_date: pd.Timestamp = None
    test_end_date: pd.Timestamp = None

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start + 1

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start + 1


class WalkForwardValidator:
    """Walk-forward cross-validation splitter for time series data.

    Produces train/test index splits where:
    - Training data always precedes test data
    - No data leakage across folds
    - Training window expands (or slides) over time

    Parameters
    ----------
    initial_train_days : int
        Minimum number of samples in the first training window.
    test_days : int
        Number of samples in each test window.
    step_days : int
        How many samples to step forward between folds.
    expanding : bool
        If True, training window expands from the start.
        If False, training window slides (fixed size).
    min_train_samples : int
        Minimum training samples required (skip fold if not met).
    """

    def __init__(
        self,
        initial_train_days: int = 504,
        test_days: int = 21,
        step_days: int = 21,
        expanding: bool = True,
        min_train_samples: int = 400,
    ):
        self.initial_train_days = initial_train_days
        self.test_days = test_days
        self.step_days = step_days
        self.expanding = expanding
        self.min_train_samples = min_train_samples

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
    ) -> List[WalkForwardFold]:
        """Generate walk-forward train/test splits.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (used for length and index).
        y : pd.Series, optional
            Labels (not used for splitting, but validated for alignment).

        Returns
        -------
        List[WalkForwardFold]
            List of fold specifications.
        """
        n = len(X)
        folds = []
        fold_num = 0

        # Determine the initial split point
        train_end = self.initial_train_days - 1
        test_start = train_end + 1

        while test_start + self.test_days <= n:
            test_end = min(test_start + self.test_days - 1, n - 1)

            # For expanding window, train always starts at 0
            # For sliding window, train starts at (test_start - initial_train_days)
            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, test_start - self.initial_train_days)

            train_size = test_start - train_start

            if train_size >= self.min_train_samples:
                fold = WalkForwardFold(
                    fold_number=fold_num,
                    train_start=train_start,
                    train_end=test_start - 1,
                    test_start=test_start,
                    test_end=test_end,
                )

                # Add dates if index is DatetimeIndex
                if isinstance(X.index, pd.DatetimeIndex):
                    fold.train_start_date = X.index[fold.train_start]
                    fold.train_end_date = X.index[fold.train_end]
                    fold.test_start_date = X.index[fold.test_start]
                    fold.test_end_date = X.index[fold.test_end]

                folds.append(fold)
                fold_num += 1

            # Step forward
            test_start += self.step_days

        logger.info(
            f"Walk-forward split: {len(folds)} folds, "
            f"initial_train={self.initial_train_days}, "
            f"test={self.test_days}, step={self.step_days}, "
            f"expanding={self.expanding}"
        )

        if folds:
            logger.info(
                f"  First fold: train {folds[0].train_size} samples, "
                f"test {folds[0].test_size} samples"
            )
            logger.info(
                f"  Last fold:  train {folds[-1].train_size} samples, "
                f"test {folds[-1].test_size} samples"
            )

        return folds

    def get_fold_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fold: WalkForwardFold,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Extract train/test data for a specific fold.

        Returns
        -------
        Tuple of (X_train, y_train, X_test, y_test)
        """
        X_train = X.iloc[fold.train_start : fold.train_end + 1]
        y_train = y.iloc[fold.train_start : fold.train_end + 1]
        X_test = X.iloc[fold.test_start : fold.test_end + 1]
        y_test = y.iloc[fold.test_start : fold.test_end + 1]

        return X_train, y_train, X_test, y_test

    def get_sklearn_splits(
        self,
        X: pd.DataFrame,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return splits in scikit-learn compatible format.

        Returns list of (train_indices, test_indices) tuples,
        compatible with sklearn's cross_val_score.
        """
        folds = self.split(X)
        splits = []

        for fold in folds:
            train_idx = np.arange(fold.train_start, fold.train_end + 1)
            test_idx = np.arange(fold.test_start, fold.test_end + 1)
            splits.append((train_idx, test_idx))

        return splits

    @classmethod
    def from_strategy_config(cls, strategy_config: dict) -> "WalkForwardValidator":
        """Create a WalkForwardValidator from a strategy config dict."""
        val_config = strategy_config.get("validation", {})
        return cls(
            initial_train_days=val_config.get("initial_train_days", 504),
            test_days=val_config.get("test_days", 21),
            step_days=val_config.get("step_days", 21),
            min_train_samples=val_config.get("min_train_samples", 400),
        )

    def summary(self, X: pd.DataFrame) -> str:
        """Return a human-readable summary of the walk-forward plan."""
        folds = self.split(X)
        lines = [
            f"Walk-Forward Validation Summary",
            f"{'=' * 40}",
            f"Total samples: {len(X)}",
            f"Number of folds: {len(folds)}",
            f"Window type: {'expanding' if self.expanding else 'sliding'}",
            f"Initial train size: {self.initial_train_days}",
            f"Test size: {self.test_days}",
            f"Step size: {self.step_days}",
            f"",
        ]

        for fold in folds:
            train_dates = ""
            test_dates = ""
            if fold.train_start_date is not None:
                train_dates = (
                    f" ({fold.train_start_date.strftime('%Y-%m-%d')} → "
                    f"{fold.train_end_date.strftime('%Y-%m-%d')})"
                )
                test_dates = (
                    f" ({fold.test_start_date.strftime('%Y-%m-%d')} → "
                    f"{fold.test_end_date.strftime('%Y-%m-%d')})"
                )

            lines.append(
                f"Fold {fold.fold_number:2d}: "
                f"train={fold.train_size:5d}{train_dates} | "
                f"test={fold.test_size:4d}{test_dates}"
            )

        return "\n".join(lines)
