"""
Regression evaluator for MarketPulse Phase 4.

Evaluates return magnitude predictions with:
- MAE, RMSE, R² (standard regression metrics)
- Directional accuracy (does the sign match?)
- Profitable-direction accuracy (practical trading metric)
- Information coefficient (rank correlation)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)


@dataclass
class RegressionFoldResult:
    """Results from a single walk-forward fold (regression)."""

    fold_number: int
    train_size: int
    test_size: int
    y_true: np.ndarray
    y_pred: np.ndarray
    test_start_date: Optional[pd.Timestamp] = None
    test_end_date: Optional[pd.Timestamp] = None

    # Computed metrics
    mae: float = 0.0
    rmse: float = 0.0
    r2: float = 0.0
    directional_accuracy: float = 0.0  # % of times sign(pred) == sign(true)
    ic: float = 0.0                     # Information Coefficient (rank corr)
    mean_pred: float = 0.0
    mean_true: float = 0.0


@dataclass
class RegressionReport:
    """Comprehensive regression evaluation report."""

    ticker: str
    strategy: str
    horizon: int
    fold_results: List[RegressionFoldResult] = field(default_factory=list)

    # Aggregate metrics
    mean_mae: float = 0.0
    mean_rmse: float = 0.0
    mean_r2: float = 0.0
    mean_directional_accuracy: float = 0.0
    mean_ic: float = 0.0
    std_mae: float = 0.0
    std_directional_accuracy: float = 0.0

    feature_importance: Optional[pd.Series] = None


class RegressionEvaluator:
    """Evaluator for return magnitude predictions.

    Computes both standard regression metrics and trading-relevant
    metrics like directional accuracy and information coefficient.
    """

    def evaluate_fold(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fold_number: int = 0,
        train_size: int = 0,
        test_start_date: Optional[pd.Timestamp] = None,
        test_end_date: Optional[pd.Timestamp] = None,
    ) -> RegressionFoldResult:
        """Evaluate a single regression fold."""
        result = RegressionFoldResult(
            fold_number=fold_number,
            train_size=train_size,
            test_size=len(y_true),
            y_true=y_true,
            y_pred=y_pred,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
        )

        # Standard regression metrics
        result.mae = mean_absolute_error(y_true, y_pred)
        result.rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        result.r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0

        # Directional accuracy: does the prediction agree on the sign?
        true_dir = np.sign(y_true)
        pred_dir = np.sign(y_pred)
        result.directional_accuracy = (true_dir == pred_dir).mean()

        # Information Coefficient (Spearman rank correlation)
        if len(y_true) >= 5:
            try:
                ic, _ = stats.spearmanr(y_true, y_pred)
                result.ic = ic if not np.isnan(ic) else 0.0
            except Exception:
                result.ic = 0.0
        else:
            result.ic = 0.0

        result.mean_pred = float(y_pred.mean())
        result.mean_true = float(y_true.mean())

        return result

    def aggregate_results(
        self,
        fold_results: List[RegressionFoldResult],
        ticker: str = "",
        strategy: str = "medium_term_regression",
        horizon: int = 5,
        feature_importance: Optional[pd.Series] = None,
    ) -> RegressionReport:
        """Aggregate across walk-forward folds."""
        report = RegressionReport(
            ticker=ticker,
            strategy=strategy,
            horizon=horizon,
            fold_results=fold_results,
            feature_importance=feature_importance,
        )

        if not fold_results:
            return report

        report.mean_mae = np.mean([f.mae for f in fold_results])
        report.mean_rmse = np.mean([f.rmse for f in fold_results])
        report.mean_r2 = np.mean([f.r2 for f in fold_results])
        report.mean_directional_accuracy = np.mean(
            [f.directional_accuracy for f in fold_results]
        )
        report.mean_ic = np.mean([f.ic for f in fold_results])
        report.std_mae = np.std([f.mae for f in fold_results])
        report.std_directional_accuracy = np.std(
            [f.directional_accuracy for f in fold_results]
        )

        return report

    def print_report(self, report: RegressionReport) -> str:
        """Formatted text report."""
        lines = [
            f"\n{'=' * 60}",
            f"MarketPulse Regression Report",
            f"{'=' * 60}",
            f"Ticker:     {report.ticker}",
            f"Strategy:   {report.strategy}",
            f"Horizon:    {report.horizon} day(s)",
            f"Folds:      {len(report.fold_results)}",
            f"",
            f"AGGREGATE METRICS",
            f"{'-' * 40}",
            f"MAE:                  {report.mean_mae:.6f} ± {report.std_mae:.6f}",
            f"RMSE:                 {report.mean_rmse:.6f}",
            f"R²:                   {report.mean_r2:.4f}",
            f"Directional Accuracy: {report.mean_directional_accuracy:.4f} ± {report.std_directional_accuracy:.4f}",
            f"Info. Coefficient:    {report.mean_ic:.4f}",
            f"",
            f"INTERPRETATION",
            f"{'-' * 40}",
        ]

        # Interpretation
        da = report.mean_directional_accuracy
        if da > 0.55:
            lines.append(f"  Direction: GOOD ({da:.1%} > 55% random baseline)")
        elif da > 0.50:
            lines.append(f"  Direction: MARGINAL ({da:.1%} ≈ coin flip)")
        else:
            lines.append(f"  Direction: POOR ({da:.1%} < 50%)")

        ic = report.mean_ic
        if ic > 0.05:
            lines.append(f"  IC: USEFUL (rank correlation = {ic:.3f})")
        elif ic > 0.02:
            lines.append(f"  IC: WEAK ({ic:.3f})")
        else:
            lines.append(f"  IC: NO SIGNAL ({ic:.3f})")

        # Per-fold
        lines.append(f"\nPER-FOLD RESULTS")
        lines.append("-" * 40)
        for f in report.fold_results:
            date_range = ""
            if f.test_start_date is not None:
                date_range = (
                    f" ({f.test_start_date.strftime('%Y-%m-%d')} → "
                    f"{f.test_end_date.strftime('%Y-%m-%d')})"
                )
            lines.append(
                f"  Fold {f.fold_number:2d}: "
                f"MAE={f.mae:.5f} DA={f.directional_accuracy:.3f} "
                f"IC={f.ic:.3f}{date_range}"
            )

        # Feature importance
        if report.feature_importance is not None:
            lines.append("")
            lines.append("TOP 10 FEATURES")
            lines.append("-" * 40)
            for feat, imp in report.feature_importance.head(10).items():
                lines.append(f"  {feat:30s} {imp:.4f}")

        text = "\n".join(lines)
        logger.info(text)
        return text

    def plot_predictions_vs_actual(
        self,
        report: RegressionReport,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Scatter plot of predicted vs actual returns."""
        all_true = np.concatenate([f.y_true for f in report.fold_results])
        all_pred = np.concatenate([f.y_pred for f in report.fold_results])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter
        ax1.scatter(all_true, all_pred, s=5, alpha=0.4, color="steelblue")
        lims = [
            min(all_true.min(), all_pred.min()),
            max(all_true.max(), all_pred.max()),
        ]
        ax1.plot(lims, lims, "r--", alpha=0.7, label="Perfect prediction")
        ax1.set_xlabel("Actual Return")
        ax1.set_ylabel("Predicted Return")
        ax1.set_title(f"Predicted vs Actual — {report.ticker}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Directional accuracy over folds
        das = [f.directional_accuracy for f in report.fold_results]
        ax2.bar(range(len(das)), das, color="steelblue", alpha=0.7)
        ax2.axhline(y=0.5, color="red", linestyle="--", label="Random (50%)")
        ax2.axhline(y=np.mean(das), color="green", linestyle="--",
                     label=f"Mean ({np.mean(das):.3f})")
        ax2.set_xlabel("Fold")
        ax2.set_ylabel("Directional Accuracy")
        ax2.set_title("Directional Accuracy per Fold")
        ax2.legend()

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
