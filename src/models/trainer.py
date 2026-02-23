"""
MarketPulse training pipeline.

Orchestrates the full workflow:
1. Load config
2. Fetch data
3. Compute features
4. Generate labels
5. Walk-forward train + evaluate
6. Produce evaluation report

Can be run as a module: python -m src.models.trainer
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.fetcher import YFinanceFetcher, create_fetcher
from ..data.market_config import load_market_config, load_strategy_config
from ..data.preprocessing import preprocess_ohlcv
from ..features.labels import generate_labels, get_clean_features_and_labels
from ..features.returns import compute_return_features
from ..features.technical import compute_technical_indicators
from ..utils.validation import WalkForwardValidator
from .evaluator import EvaluationReport, MarketPulseEvaluator
from .xgboost_classifier import MarketPulseXGBClassifier

# Sentiment (Phase 2) — optional import
try:
    from ..features.sentiment import fetch_and_score_news, merge_sentiment_features
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

logger = logging.getLogger(__name__)


class MarketPulseTrainer:
    """End-to-end training pipeline for MarketPulse.

    Parameters
    ----------
    market_name : str
        Market config to use (e.g. 'stocks', 'crypto').
    strategy_name : str
        Strategy config to use (e.g. 'short_term').
    tickers : list, optional
        Override default tickers from market config.
    """

    def __init__(
        self,
        market_name: str = "stocks",
        strategy_name: str = "short_term",
        tickers: Optional[List[str]] = None,
        use_sentiment: bool = False,
        newsapi_key: Optional[str] = None,
        sentiment_days: int = 60,
    ):
        # Load configs
        self.market_config = load_market_config(market_name)
        self.strategy_config = load_strategy_config(strategy_name)

        # Override tickers if provided
        self.base_tickers = tickers or self.market_config.default_tickers
        self.formatted_tickers = self.market_config.format_tickers(self.base_tickers)

        # Initialize components
        self.fetcher = create_fetcher(
            market_config=self.market_config,
            source=self.market_config.data_source,
        )
        self.validator = WalkForwardValidator.from_strategy_config(
            self.strategy_config
        )
        self.evaluator = MarketPulseEvaluator(
            num_classes=self.strategy_config.get("num_classes", 3)
        )

        # Strategy settings
        self.horizon = self.strategy_config.get("default_horizon", 1)
        self.num_classes = self.strategy_config.get("num_classes", 3)
        self.threshold = self.strategy_config.get("threshold", 0.01)
        self.label_type = self.strategy_config.get("label_type", "classification")
        self.years_of_history = self.strategy_config.get("data", {}).get(
            "years_of_history", 5
        )

        # Results storage
        self.reports: Dict[str, EvaluationReport] = {}
        self.models: Dict[str, MarketPulseXGBClassifier] = {}

        # Sentiment settings (Phase 2)
        self.use_sentiment = use_sentiment and SENTIMENT_AVAILABLE
        self.newsapi_key = newsapi_key
        self.sentiment_days = sentiment_days
        if use_sentiment and not SENTIMENT_AVAILABLE:
            logger.warning(
                "Sentiment requested but transformers/torch not installed. "
                "Install with: pip install transformers torch"
            )

    def run(self, verbose: bool = True) -> Dict[str, EvaluationReport]:
        """Execute the full training pipeline for all tickers.

        Returns
        -------
        Dict[str, EvaluationReport]
            Mapping of ticker -> evaluation report.
        """
        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )

        logger.info(
            f"Starting MarketPulse training pipeline\n"
            f"  Market: {self.market_config.display_name}\n"
            f"  Strategy: {self.strategy_config.get('display_name', self.strategy_config.get('name'))}\n"
            f"  Tickers: {len(self.formatted_tickers)}\n"
            f"  Horizon: {self.horizon} day(s)\n"
            f"  Label type: {self.label_type} ({self.num_classes} classes)\n"
        )

        # Date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (
            datetime.now() - timedelta(days=self.years_of_history * 365)
        ).strftime("%Y-%m-%d")

        for i, (base_ticker, fmt_ticker) in enumerate(
            zip(self.base_tickers, self.formatted_tickers)
        ):
            logger.info(
                f"\n{'=' * 50}\n"
                f"[{i + 1}/{len(self.formatted_tickers)}] Processing {fmt_ticker}\n"
                f"{'=' * 50}"
            )

            try:
                report = self._train_single_ticker(
                    base_ticker, fmt_ticker, start_date, end_date
                )
                self.reports[fmt_ticker] = report

                # Print summary
                self.evaluator.print_report(report)

            except Exception as e:
                logger.error(f"Failed to process {fmt_ticker}: {e}", exc_info=True)

        # Summary
        self._print_summary()

        return self.reports

    def _train_single_ticker(
        self,
        base_ticker: str,
        fmt_ticker: str,
        start_date: str,
        end_date: str,
    ) -> EvaluationReport:
        """Full pipeline for a single ticker."""

        # 1. Fetch data
        logger.info(f"Fetching data for {fmt_ticker}...")
        raw_df = self.fetcher.fetch(fmt_ticker, start=start_date, end=end_date)
        if raw_df.empty:
            raise ValueError(f"No data returned for {fmt_ticker}")
        logger.info(f"  Fetched {len(raw_df)} rows")

        # 2. Preprocess
        logger.info("Preprocessing...")
        df = preprocess_ohlcv(raw_df, market_config=self.market_config)
        logger.info(f"  {len(df)} rows after preprocessing")

        # 3. Compute features
        logger.info("Computing technical indicators...")
        df = compute_technical_indicators(df)

        logger.info("Computing return features...")
        df = compute_return_features(df)

        # 3b. Sentiment features (Phase 2)
        if self.use_sentiment:
            logger.info("Fetching & scoring news sentiment...")
            try:
                _, daily_sentiment = fetch_and_score_news(
                    ticker=base_ticker,
                    newsapi_key=self.newsapi_key,
                    days_back=self.sentiment_days,
                )
                df = merge_sentiment_features(df, daily_sentiment)
                logger.info(f"  Added {len(daily_sentiment.columns)} sentiment features")
            except Exception as e:
                logger.warning(f"Sentiment failed for {base_ticker}: {e}")

        # 4. Generate labels
        logger.info(f"Generating labels (horizon={self.horizon}d)...")
        df = generate_labels(
            df,
            horizon=self.horizon,
            label_type=self.label_type,
            num_classes=self.num_classes,
            threshold=self.threshold,
        )

        # 5. Extract clean features and labels
        X, y = get_clean_features_and_labels(df)
        logger.info(f"  Clean dataset: {len(X)} samples, {X.shape[1]} features")

        if len(X) < self.validator.initial_train_days + self.validator.test_days:
            raise ValueError(
                f"Insufficient data for walk-forward validation: "
                f"{len(X)} samples < {self.validator.initial_train_days} + "
                f"{self.validator.test_days} required"
            )

        # 6. Walk-forward training and evaluation
        logger.info("Starting walk-forward validation...")
        folds = self.validator.split(X)
        fold_results = []
        last_model = None

        for fold in folds:
            X_train, y_train, X_test, y_test = self.validator.get_fold_data(
                X, y, fold
            )

            # Train model
            model = MarketPulseXGBClassifier.from_strategy_config(
                self.strategy_config
            )
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            # Evaluate fold
            fold_result = self.evaluator.evaluate_fold(
                y_true=y_test.values.astype(int),
                y_pred=y_pred,
                y_proba=y_proba,
                fold_number=fold.fold_number,
                train_size=fold.train_size,
                test_start_date=fold.test_start_date,
                test_end_date=fold.test_end_date,
            )
            fold_results.append(fold_result)

            logger.info(
                f"  Fold {fold.fold_number:2d}: "
                f"acc={fold_result.accuracy:.3f} "
                f"f1={fold_result.f1:.3f}"
            )

            last_model = model

        # Store the last trained model (most recent data)
        self.models[fmt_ticker] = last_model

        # Get feature importance from last model
        feature_importance = last_model.get_feature_importance()

        # Aggregate results
        report = self.evaluator.aggregate_results(
            fold_results=fold_results,
            ticker=fmt_ticker,
            strategy=self.strategy_config.get("name", "short_term"),
            horizon=self.horizon,
            feature_importance=feature_importance,
        )

        return report

    def predict_latest(self, ticker: str) -> Optional[Dict]:
        """Get the latest prediction for a ticker.

        Uses the most recently trained model to predict the next move.

        Returns
        -------
        dict or None
            {
                'ticker': str,
                'prediction': str ('UP', 'DOWN', 'FLAT'),
                'confidence': float,
                'probabilities': dict,
                'date': str,
            }
        """
        fmt_ticker = self.market_config.format_ticker(ticker)

        if fmt_ticker not in self.models:
            logger.warning(f"No trained model for {fmt_ticker}")
            return None

        model = self.models[fmt_ticker]

        # Fetch latest data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")

        raw_df = self.fetcher.fetch(fmt_ticker, start=start_date, end=end_date)
        if raw_df.empty:
            return None

        df = preprocess_ohlcv(raw_df, market_config=self.market_config)
        df = compute_technical_indicators(df)
        df = compute_return_features(df)

        # Get the last row of features (current state)
        exclude_cols = {
            "label", "label_name",
            "open", "high", "low", "close", "volume", "adj_close",
        }
        exclude_cols.update(col for col in df.columns if col.startswith("fwd_return"))

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Get last valid row
        X_latest = df[feature_cols].dropna().iloc[[-1]]

        if X_latest.empty:
            return None

        # Predict
        pred = model.predict(X_latest)[0]
        proba = model.predict_proba(X_latest)[0]

        class_names = {0: "DOWN", 1: "FLAT", 2: "UP"}
        if self.num_classes == 2:
            class_names = {0: "DOWN", 1: "UP"}

        return {
            "ticker": fmt_ticker,
            "prediction": class_names.get(int(pred), "UNKNOWN"),
            "confidence": float(proba.max()),
            "probabilities": {
                class_names[i]: float(p) for i, p in enumerate(proba)
            },
            "date": X_latest.index[-1].strftime("%Y-%m-%d"),
        }

    def _print_summary(self):
        """Print aggregate summary across all tickers."""
        if not self.reports:
            logger.info("No results to summarize.")
            return

        lines = [
            f"\n{'=' * 60}",
            f"OVERALL SUMMARY",
            f"{'=' * 60}",
            f"{'Ticker':<12} {'Accuracy':>10} {'F1':>10} {'Baseline':>10} {'Lift':>10}",
            f"{'-' * 52}",
        ]

        for ticker, report in self.reports.items():
            lift = report.mean_accuracy - report.baseline_accuracy
            lines.append(
                f"{ticker:<12} "
                f"{report.mean_accuracy:>10.4f} "
                f"{report.mean_f1:>10.4f} "
                f"{report.baseline_accuracy:>10.4f} "
                f"{lift:>+10.4f}"
            )

        avg_acc = np.mean([r.mean_accuracy for r in self.reports.values()])
        avg_f1 = np.mean([r.mean_f1 for r in self.reports.values()])
        lines.extend([
            f"{'-' * 52}",
            f"{'AVERAGE':<12} {avg_acc:>10.4f} {avg_f1:>10.4f}",
        ])

        text = "\n".join(lines)
        logger.info(text)
        print(text)


# ──────────────────── CLI Entry Point ────────────────────

def main():
    """Run the training pipeline from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="MarketPulse Training Pipeline")
    parser.add_argument(
        "--market", type=str, default="stocks", help="Market name (default: stocks)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="short_term",
        help="Strategy name (default: short_term)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="+",
        default=None,
        help="Override tickers (space-separated base symbols)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Override prediction horizon (days)",
    )
    parser.add_argument(
        "--sentiment",
        action="store_true",
        help="Enable sentiment features (Phase 2)",
    )
    parser.add_argument(
        "--newsapi-key",
        type=str,
        default=None,
        help="NewsAPI key for richer news data",
    )

    args = parser.parse_args()

    trainer = MarketPulseTrainer(
        market_name=args.market,
        strategy_name=args.strategy,
        tickers=args.tickers,
        use_sentiment=args.sentiment,
        newsapi_key=args.newsapi_key,
    )

    if args.horizon:
        trainer.horizon = args.horizon

    trainer.run(verbose=True)


if __name__ == "__main__":
    main()
