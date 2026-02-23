"""
Model evaluator for MarketPulse.

Computes classification metrics, generates visualizations,
and produces comprehensive evaluation reports.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""

    fold_number: int
    train_size: int
    test_size: int
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray] = None
    test_start_date: Optional[pd.Timestamp] = None
    test_end_date: Optional[pd.Timestamp] = None

    # Computed metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report across all walk-forward folds."""

    ticker: str
    strategy: str
    horizon: int
    fold_results: List[FoldResult] = field(default_factory=list)

    # Aggregate metrics
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    mean_f1: float = 0.0
    std_f1: float = 0.0
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_roc_auc: float = 0.0
    baseline_accuracy: float = 0.0  # majority class baseline

    feature_importance: Optional[pd.Series] = None


class MarketPulseEvaluator:
    """Evaluator that computes metrics and generates visualizations.

    Handles both binary and multi-class classification,
    computes per-fold and aggregate metrics, and creates publication-quality plots.
    """

    CLASS_NAMES_2 = {0: "DOWN", 1: "UP"}
    CLASS_NAMES_3 = {0: "DOWN", 1: "FLAT", 2: "UP"}

    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.class_names = (
            self.CLASS_NAMES_3 if num_classes == 3 else self.CLASS_NAMES_2
        )

    def evaluate_fold(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        fold_number: int = 0,
        train_size: int = 0,
        test_start_date: Optional[pd.Timestamp] = None,
        test_end_date: Optional[pd.Timestamp] = None,
    ) -> FoldResult:
        """Evaluate predictions for a single walk-forward fold.

        Returns
        -------
        FoldResult
            Metrics and predictions for this fold.
        """
        result = FoldResult(
            fold_number=fold_number,
            train_size=train_size,
            test_size=len(y_true),
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
        )

        # Core metrics
        result.accuracy = accuracy_score(y_true, y_pred)
        result.precision = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        result.recall = recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        result.f1 = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )

        # ROC AUC (requires probability predictions)
        if y_proba is not None:
            try:
                if self.num_classes == 2:
                    result.roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    result.roc_auc = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted"
                    )
            except (ValueError, IndexError) as e:
                logger.warning(f"ROC AUC calculation failed for fold {fold_number}: {e}")
                result.roc_auc = 0.0

        return result

    def aggregate_results(
        self,
        fold_results: List[FoldResult],
        ticker: str = "",
        strategy: str = "short_term",
        horizon: int = 1,
        feature_importance: Optional[pd.Series] = None,
    ) -> EvaluationReport:
        """Aggregate results across all walk-forward folds.

        Returns
        -------
        EvaluationReport
            Comprehensive report with aggregate metrics.
        """
        report = EvaluationReport(
            ticker=ticker,
            strategy=strategy,
            horizon=horizon,
            fold_results=fold_results,
            feature_importance=feature_importance,
        )

        if not fold_results:
            return report

        accuracies = [f.accuracy for f in fold_results]
        f1s = [f.f1 for f in fold_results]

        report.mean_accuracy = np.mean(accuracies)
        report.std_accuracy = np.std(accuracies)
        report.mean_f1 = np.mean(f1s)
        report.std_f1 = np.std(f1s)
        report.mean_precision = np.mean([f.precision for f in fold_results])
        report.mean_recall = np.mean([f.recall for f in fold_results])
        report.mean_roc_auc = np.mean([f.roc_auc for f in fold_results])

        # Baseline: majority class accuracy
        all_true = np.concatenate([f.y_true for f in fold_results])
        if len(all_true) > 0:
            majority_class = pd.Series(all_true).mode().iloc[0]
            report.baseline_accuracy = (all_true == majority_class).mean()

        return report

    def print_report(self, report: EvaluationReport) -> str:
        """Generate a formatted text report."""
        lines = [
            f"\n{'=' * 60}",
            f"MarketPulse Evaluation Report",
            f"{'=' * 60}",
            f"Ticker:     {report.ticker}",
            f"Strategy:   {report.strategy}",
            f"Horizon:    {report.horizon} day(s)",
            f"Folds:      {len(report.fold_results)}",
            f"",
            f"AGGREGATE METRICS",
            f"{'-' * 40}",
            f"Accuracy:   {report.mean_accuracy:.4f} ± {report.std_accuracy:.4f}",
            f"F1 Score:   {report.mean_f1:.4f} ± {report.std_f1:.4f}",
            f"Precision:  {report.mean_precision:.4f}",
            f"Recall:     {report.mean_recall:.4f}",
            f"ROC AUC:    {report.mean_roc_auc:.4f}",
            f"",
            f"Baseline (majority class): {report.baseline_accuracy:.4f}",
            f"Lift over baseline:        {report.mean_accuracy - report.baseline_accuracy:+.4f}",
            f"",
        ]

        # Per-fold breakdown
        lines.append("PER-FOLD RESULTS")
        lines.append("-" * 40)
        for fold in report.fold_results:
            date_range = ""
            if fold.test_start_date is not None:
                date_range = (
                    f" ({fold.test_start_date.strftime('%Y-%m-%d')} → "
                    f"{fold.test_end_date.strftime('%Y-%m-%d')})"
                )
            lines.append(
                f"  Fold {fold.fold_number:2d}: "
                f"acc={fold.accuracy:.3f} f1={fold.f1:.3f} "
                f"[train={fold.train_size}, test={fold.test_size}]"
                f"{date_range}"
            )

        # Feature importance
        if report.feature_importance is not None:
            lines.append("")
            lines.append("TOP 10 FEATURES (by importance)")
            lines.append("-" * 40)
            for feat, imp in report.feature_importance.head(10).items():
                lines.append(f"  {feat:30s} {imp:.4f}")

        text = "\n".join(lines)
        logger.info(text)
        return text

    def plot_fold_accuracy(
        self,
        report: EvaluationReport,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot accuracy over time (one point per fold)."""
        fig, ax = plt.subplots(figsize=(12, 5))

        fold_nums = [f.fold_number for f in report.fold_results]
        accuracies = [f.accuracy for f in report.fold_results]

        ax.plot(fold_nums, accuracies, "b-o", label="Fold Accuracy", markersize=4)
        ax.axhline(
            y=report.mean_accuracy,
            color="green",
            linestyle="--",
            label=f"Mean Accuracy ({report.mean_accuracy:.3f})",
        )
        ax.axhline(
            y=report.baseline_accuracy,
            color="red",
            linestyle=":",
            label=f"Baseline ({report.baseline_accuracy:.3f})",
        )

        ax.set_xlabel("Fold Number")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"MarketPulse Walk-Forward Accuracy — {report.ticker}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_confusion_matrix(
        self,
        report: EvaluationReport,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot aggregate confusion matrix across all folds."""
        all_true = np.concatenate([f.y_true for f in report.fold_results])
        all_pred = np.concatenate([f.y_pred for f in report.fold_results])

        cm = confusion_matrix(all_true, all_pred)
        labels = [self.class_names[i] for i in sorted(self.class_names.keys())]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(
            f"Confusion Matrix — {report.ticker} "
            f"({len(report.fold_results)} folds)"
        )

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_feature_importance(
        self,
        report: EvaluationReport,
        top_n: int = 15,
        save_path: Optional[str] = None,
    ) -> Optional[plt.Figure]:
        """Plot top feature importances."""
        if report.feature_importance is None:
            return None

        top = report.feature_importance.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 6))
        top.sort_values().plot(kind="barh", ax=ax, color="steelblue")
        ax.set_xlabel("Importance (Gain)")
        ax.set_title(f"Top {top_n} Features — {report.ticker}")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
