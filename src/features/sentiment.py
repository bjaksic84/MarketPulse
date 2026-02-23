"""
Sentiment analysis for MarketPulse Phase 2.

Uses FinBERT (ProsusAI/finbert) — a BERT model fine-tuned on
60,000 financial news sentences. Outperforms general-purpose
sentiment models on financial text.

Output: per-headline sentiment score + aggregated daily features.

Features produced per day:
- sentiment_mean: average sentiment (-1 to +1)
- sentiment_std: sentiment dispersion
- sentiment_positive_ratio: fraction of positive headlines
- sentiment_negative_ratio: fraction of negative headlines
- sentiment_count: number of headlines
- sentiment_momentum_3d: 3-day rolling mean of sentiment_mean
- sentiment_momentum_5d: 5-day rolling mean of sentiment_mean
- sentiment_surprise: today's sentiment - 5d rolling mean (contrarian signal)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────── FinBERT Scorer ───────────────


class FinBERTSentimentScorer:
    """Score financial headlines using FinBERT.

    Loads model lazily on first call. Model is ~420 MB downloaded once
    and cached by HuggingFace.

    Parameters
    ----------
    model_name : str
        HuggingFace model name. Default: 'ProsusAI/finbert'.
    device : str
        'cpu' or 'cuda'. Default: auto-detect.
    batch_size : int
        Headlines per batch. Lower = less memory.
    """

    # Maps FinBERT output to numeric score
    LABEL_MAP = {
        "positive": 1.0,
        "negative": -1.0,
        "neutral": 0.0,
    }

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: Optional[str] = None,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._pipeline = None

        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def _load_model(self):
        """Lazy-load the FinBERT pipeline."""
        if self._pipeline is not None:
            return

        logger.info(f"Loading FinBERT model ({self.model_name})...")
        from transformers import pipeline

        self._pipeline = pipeline(
            "text-classification",
            model=self.model_name,
            device=0 if self.device == "cuda" else -1,
            truncation=True,
            max_length=512,
        )
        logger.info(f"FinBERT loaded on {self.device}")

    def score_headlines(
        self,
        headlines: List[str],
    ) -> List[Dict]:
        """Score a list of headlines.

        Parameters
        ----------
        headlines : list of str
            Financial news headlines or sentences.

        Returns
        -------
        list of dict
            Each dict: {'headline': str, 'label': str, 'score': float,
                        'confidence': float}
            score ∈ [-1, +1], confidence ∈ [0, 1]
        """
        self._load_model()

        if not headlines:
            return []

        # Clean headlines
        clean = [
            h.strip()[:512] for h in headlines
            if h and isinstance(h, str) and len(h.strip()) > 5
        ]

        if not clean:
            return []

        # Batch inference
        results = []
        for i in range(0, len(clean), self.batch_size):
            batch = clean[i : i + self.batch_size]
            preds = self._pipeline(batch)

            for headline, pred in zip(batch, preds):
                label = pred["label"].lower()
                results.append({
                    "headline": headline,
                    "label": label,
                    "score": self.LABEL_MAP.get(label, 0.0),
                    "confidence": pred["score"],
                })

        logger.info(
            f"Scored {len(results)} headlines: "
            f"{sum(1 for r in results if r['label'] == 'positive')} pos, "
            f"{sum(1 for r in results if r['label'] == 'negative')} neg, "
            f"{sum(1 for r in results if r['label'] == 'neutral')} neutral"
        )

        return results

    def score_dataframe(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Score all headlines in a news DataFrame.

        Parameters
        ----------
        news_df : pd.DataFrame
            Must have 'title' column. Optionally 'published'.

        Returns
        -------
        pd.DataFrame
            Original columns + 'sentiment_label', 'sentiment_score', 'confidence'
        """
        if news_df.empty or "title" not in news_df.columns:
            return news_df

        headlines = news_df["title"].tolist()
        scored = self.score_headlines(headlines)

        if not scored:
            news_df["sentiment_label"] = None
            news_df["sentiment_score"] = 0.0
            news_df["confidence"] = 0.0
            return news_df

        # Align scores back to dataframe
        scores_df = pd.DataFrame(scored)
        # Match by index (same order as input)
        for col in ["label", "score", "confidence"]:
            target = f"sentiment_{col}" if col != "confidence" else col
            if col == "label":
                target = "sentiment_label"
            news_df[target] = scores_df[col].values[: len(news_df)]

        return news_df


# ─────────────── Daily Feature Aggregation ───────────────


def aggregate_daily_sentiment(
    scored_news: pd.DataFrame,
    date_col: str = "published",
) -> pd.DataFrame:
    """Aggregate per-headline scores into daily sentiment features.

    Parameters
    ----------
    scored_news : pd.DataFrame
        Must have 'published' (datetime) and 'sentiment_score' columns.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, one row per trading day, with sentiment features.
    """
    if scored_news.empty or "sentiment_score" not in scored_news.columns:
        return pd.DataFrame()

    df = scored_news.copy()
    df["date"] = pd.to_datetime(df[date_col]).dt.normalize()

    daily = df.groupby("date").agg(
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_std=("sentiment_score", "std"),
        sentiment_count=("sentiment_score", "count"),
        sentiment_positive_ratio=(
            "sentiment_score", lambda x: (x > 0).mean()
        ),
        sentiment_negative_ratio=(
            "sentiment_score", lambda x: (x < 0).mean()
        ),
    )

    # Fill NaN std (single-headline days) with 0
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0)

    # Rolling features
    daily["sentiment_momentum_3d"] = (
        daily["sentiment_mean"].rolling(3, min_periods=1).mean()
    )
    daily["sentiment_momentum_5d"] = (
        daily["sentiment_mean"].rolling(5, min_periods=1).mean()
    )
    daily["sentiment_surprise"] = (
        daily["sentiment_mean"] - daily["sentiment_momentum_5d"]
    )

    daily.index = pd.DatetimeIndex(daily.index)
    daily.index.name = "date"

    logger.info(
        f"Daily sentiment features: {len(daily)} days, "
        f"mean={daily['sentiment_mean'].mean():.3f}"
    )

    return daily


def merge_sentiment_features(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge daily sentiment features into the price/feature DataFrame.

    Uses forward-fill to handle weekends/holidays where no news
    is published but markets are open (or vice versa).

    Parameters
    ----------
    price_df : pd.DataFrame
        OHLCV + technical features with DatetimeIndex.
    sentiment_df : pd.DataFrame
        Daily sentiment features from aggregate_daily_sentiment().

    Returns
    -------
    pd.DataFrame
        price_df with sentiment columns added.
    """
    if sentiment_df.empty:
        logger.warning("Empty sentiment DataFrame, adding zero-filled columns")
        for col in [
            "sentiment_mean", "sentiment_std", "sentiment_count",
            "sentiment_positive_ratio", "sentiment_negative_ratio",
            "sentiment_momentum_3d", "sentiment_momentum_5d",
            "sentiment_surprise",
        ]:
            price_df[col] = 0.0
        return price_df

    # Ensure both indices are normalized dates
    price_df.index = pd.DatetimeIndex(price_df.index).normalize()
    sentiment_df.index = pd.DatetimeIndex(sentiment_df.index).normalize()

    # Join (left join on price dates)
    merged = price_df.join(sentiment_df, how="left")

    # Forward-fill: weekend news carries into Monday
    sentiment_cols = sentiment_df.columns.tolist()
    merged[sentiment_cols] = merged[sentiment_cols].ffill()

    # Fill remaining NaNs (start of series) with neutral
    merged["sentiment_mean"] = merged["sentiment_mean"].fillna(0.0)
    merged["sentiment_std"] = merged["sentiment_std"].fillna(0.0)
    merged["sentiment_count"] = merged["sentiment_count"].fillna(0)
    merged["sentiment_positive_ratio"] = merged["sentiment_positive_ratio"].fillna(0.33)
    merged["sentiment_negative_ratio"] = merged["sentiment_negative_ratio"].fillna(0.33)
    merged["sentiment_momentum_3d"] = merged["sentiment_momentum_3d"].fillna(0.0)
    merged["sentiment_momentum_5d"] = merged["sentiment_momentum_5d"].fillna(0.0)
    merged["sentiment_surprise"] = merged["sentiment_surprise"].fillna(0.0)

    logger.info(f"Merged {len(sentiment_cols)} sentiment features into price data")

    return merged


# ─────────────── Convenience: Full Pipeline ───────────────

def fetch_and_score_news(
    ticker: str,
    company_name: Optional[str] = None,
    newsapi_key: Optional[str] = None,
    days_back: int = 30,
    device: str = "cpu",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end: fetch news → score with FinBERT → aggregate daily.

    Parameters
    ----------
    ticker : str
        Stock ticker (e.g. 'AAPL').
    company_name : str, optional
        Full name for broader search (e.g. 'Apple'). If None, uses ticker.
    newsapi_key : str, optional
        NewsAPI key for richer data.
    days_back : int
        How many days of news history to fetch.
    device : str
        'cpu' or 'cuda' for FinBERT.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (scored_headlines_df, daily_sentiment_df)
    """
    from .news_fetcher import AggregateNewsFetcher

    query = company_name or ticker

    start_date = (
        pd.Timestamp.now() - pd.Timedelta(days=days_back)
    ).strftime("%Y-%m-%d")
    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    # Fetch
    fetcher = AggregateNewsFetcher(newsapi_key=newsapi_key)
    news_df = fetcher.fetch_all(
        query=query,
        start_date=start_date,
        end_date=end_date,
    )

    if news_df.empty:
        logger.warning(f"No news found for {query}")
        return news_df, pd.DataFrame()

    # Score
    scorer = FinBERTSentimentScorer(device=device)
    scored_df = scorer.score_dataframe(news_df)

    # Aggregate
    daily_df = aggregate_daily_sentiment(scored_df)

    return scored_df, daily_df
