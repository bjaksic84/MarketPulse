"""
Tests for sentiment analysis module (Phase 2).

Tests the scoring, aggregation, and merging logic
using mock/synthetic data (no real API calls or model downloads).
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


# ─────────────── Test aggregate_daily_sentiment ───────────────

class TestAggregateDailySentiment:
    def _make_scored_df(self, n_headlines=20, n_days=5):
        """Create synthetic scored news DataFrame."""
        np.random.seed(42)
        base_date = datetime(2024, 1, 15)
        records = []
        for i in range(n_headlines):
            day = np.random.randint(0, n_days)
            records.append({
                "title": f"Headline {i}",
                "published": base_date + timedelta(days=day, hours=np.random.randint(8, 18)),
                "sentiment_score": np.random.choice([-1.0, 0.0, 1.0], p=[0.3, 0.3, 0.4]),
                "sentiment_label": np.random.choice(["negative", "neutral", "positive"]),
                "confidence": np.random.uniform(0.6, 0.99),
            })
        return pd.DataFrame(records)

    def test_returns_dataframe(self):
        from src.features.sentiment import aggregate_daily_sentiment

        scored = self._make_scored_df()
        daily = aggregate_daily_sentiment(scored)
        assert isinstance(daily, pd.DataFrame)

    def test_has_expected_columns(self):
        from src.features.sentiment import aggregate_daily_sentiment

        scored = self._make_scored_df()
        daily = aggregate_daily_sentiment(scored)

        expected_cols = [
            "sentiment_mean", "sentiment_std", "sentiment_count",
            "sentiment_positive_ratio", "sentiment_negative_ratio",
            "sentiment_momentum_3d", "sentiment_momentum_5d",
            "sentiment_surprise",
        ]
        for col in expected_cols:
            assert col in daily.columns, f"Missing column: {col}"

    def test_count_matches_input(self):
        from src.features.sentiment import aggregate_daily_sentiment

        scored = self._make_scored_df(n_headlines=20, n_days=3)
        daily = aggregate_daily_sentiment(scored)
        assert daily["sentiment_count"].sum() == 20

    def test_mean_in_range(self):
        from src.features.sentiment import aggregate_daily_sentiment

        scored = self._make_scored_df()
        daily = aggregate_daily_sentiment(scored)
        assert daily["sentiment_mean"].min() >= -1.0
        assert daily["sentiment_mean"].max() <= 1.0

    def test_ratio_sums(self):
        from src.features.sentiment import aggregate_daily_sentiment

        scored = self._make_scored_df(n_headlines=100, n_days=5)
        daily = aggregate_daily_sentiment(scored)
        # Pos + neg + neutral ≈ 1.0
        for _, row in daily.iterrows():
            total = row["sentiment_positive_ratio"] + row["sentiment_negative_ratio"]
            assert total <= 1.0 + 1e-9  # pos + neg can't exceed 1

    def test_empty_input(self):
        from src.features.sentiment import aggregate_daily_sentiment

        empty = pd.DataFrame(columns=["published", "sentiment_score"])
        result = aggregate_daily_sentiment(empty)
        assert result.empty


# ─────────────── Test merge_sentiment_features ───────────────

class TestMergeSentimentFeatures:
    def test_merge_adds_columns(self):
        from src.features.sentiment import merge_sentiment_features

        # Price DataFrame
        dates = pd.date_range("2024-01-15", periods=10, freq="B")
        price_df = pd.DataFrame(
            {"close": np.random.randn(10).cumsum() + 100},
            index=dates,
        )

        # Sentiment DataFrame
        sent_dates = pd.date_range("2024-01-15", periods=5, freq="B")
        sent_df = pd.DataFrame(
            {
                "sentiment_mean": np.random.randn(5) * 0.3,
                "sentiment_std": np.abs(np.random.randn(5)) * 0.2,
                "sentiment_count": np.random.randint(1, 20, 5),
                "sentiment_positive_ratio": np.random.uniform(0.2, 0.6, 5),
                "sentiment_negative_ratio": np.random.uniform(0.1, 0.4, 5),
                "sentiment_momentum_3d": np.random.randn(5) * 0.1,
                "sentiment_momentum_5d": np.random.randn(5) * 0.1,
                "sentiment_surprise": np.random.randn(5) * 0.05,
            },
            index=sent_dates,
        )

        merged = merge_sentiment_features(price_df.copy(), sent_df)
        assert "sentiment_mean" in merged.columns
        assert len(merged) == len(price_df)

    def test_empty_sentiment_adds_zeros(self):
        from src.features.sentiment import merge_sentiment_features

        dates = pd.date_range("2024-01-15", periods=5, freq="B")
        price_df = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104]},
            index=dates,
        )
        empty_sent = pd.DataFrame()

        merged = merge_sentiment_features(price_df.copy(), empty_sent)
        assert "sentiment_mean" in merged.columns
        assert (merged["sentiment_mean"] == 0.0).all()

    def test_no_nans_after_merge(self):
        from src.features.sentiment import merge_sentiment_features

        dates = pd.date_range("2024-01-15", periods=10, freq="B")
        price_df = pd.DataFrame(
            {"close": np.random.randn(10).cumsum() + 100},
            index=dates,
        )

        sent_dates = pd.date_range("2024-01-17", periods=3, freq="B")
        sent_df = pd.DataFrame(
            {
                "sentiment_mean": [0.5, -0.3, 0.1],
                "sentiment_std": [0.2, 0.1, 0.3],
                "sentiment_count": [5, 3, 7],
                "sentiment_positive_ratio": [0.6, 0.2, 0.5],
                "sentiment_negative_ratio": [0.2, 0.6, 0.3],
                "sentiment_momentum_3d": [0.3, 0.1, 0.2],
                "sentiment_momentum_5d": [0.2, 0.0, 0.1],
                "sentiment_surprise": [0.1, -0.3, 0.0],
            },
            index=sent_dates,
        )

        merged = merge_sentiment_features(price_df.copy(), sent_df)
        sentiment_cols = [c for c in merged.columns if "sentiment" in c]
        assert not merged[sentiment_cols].isna().any().any()


# ─────────────── Test FinBERT scorer (mocked) ───────────────

class TestFinBERTScorerMocked:
    @patch("src.features.sentiment.FinBERTSentimentScorer._load_model")
    def test_score_headlines_empty(self, mock_load):
        from src.features.sentiment import FinBERTSentimentScorer

        scorer = FinBERTSentimentScorer()
        result = scorer.score_headlines([])
        assert result == []

    @patch("src.features.sentiment.FinBERTSentimentScorer._load_model")
    def test_score_headlines_filters_short(self, mock_load):
        from src.features.sentiment import FinBERTSentimentScorer

        scorer = FinBERTSentimentScorer()
        # Mock the pipeline
        scorer._pipeline = MagicMock(return_value=[])
        result = scorer.score_headlines(["hi", "x", ""])  # All too short
        assert result == []

    def test_label_map_values(self):
        from src.features.sentiment import FinBERTSentimentScorer

        assert FinBERTSentimentScorer.LABEL_MAP["positive"] == 1.0
        assert FinBERTSentimentScorer.LABEL_MAP["negative"] == -1.0
        assert FinBERTSentimentScorer.LABEL_MAP["neutral"] == 0.0


# ─────────────── Test News Fetchers (structure) ───────────────

class TestNewsFetcherStructure:
    def test_yfinance_fetcher_instantiates(self):
        from src.data.news_fetcher import YFinanceNewsFetcher
        fetcher = YFinanceNewsFetcher()
        assert fetcher is not None

    def test_gdelt_fetcher_instantiates(self):
        from src.data.news_fetcher import GDELTNewsFetcher
        fetcher = GDELTNewsFetcher()
        assert fetcher is not None

    def test_aggregate_fetcher_instantiates(self):
        from src.data.news_fetcher import AggregateNewsFetcher
        fetcher = AggregateNewsFetcher(newsapi_key=None, use_gdelt=False)
        assert len(fetcher.sources) == 1  # only yfinance

    def test_aggregate_fetcher_with_all_sources(self):
        from src.data.news_fetcher import AggregateNewsFetcher
        fetcher = AggregateNewsFetcher(newsapi_key="test", use_gdelt=True)
        assert len(fetcher.sources) == 3  # yfinance + newsapi + gdelt

    def test_make_record(self):
        from src.data.news_fetcher import _make_record
        record = _make_record(
            title="Test headline",
            source="test",
            published=datetime(2024, 1, 15),
            ticker="AAPL",
        )
        assert record["title"] == "Test headline"
        assert record["ticker"] == "AAPL"
        assert isinstance(record["published"], pd.Timestamp)
