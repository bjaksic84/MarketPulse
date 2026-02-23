"""
News data fetcher for MarketPulse Phase 2.

Fetches financial headlines from multiple free sources:
1. yfinance built-in news (always available, no API key)
2. NewsAPI (free tier: 100 req/day, 1-month history)
3. GDELT API (unlimited, global news, no key needed)

Each source returns standardized headline records:
{
    "title": str,
    "source": str,
    "published": datetime,
    "url": str,
    "ticker": str,
}
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────── Standardized Record ───────────────

def _make_record(
    title: str,
    source: str,
    published: datetime,
    url: str = "",
    ticker: str = "",
    description: str = "",
) -> Dict:
    return {
        "title": title,
        "source": source,
        "published": pd.Timestamp(published),
        "url": url,
        "ticker": ticker,
        "description": description,
    }


# ─────────────── Abstract Base ───────────────

class NewsFetcher(ABC):
    """Abstract base for news data sources."""

    @abstractmethod
    def fetch_news(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: int = 100,
    ) -> List[Dict]:
        """Fetch news articles matching a query.

        Returns list of standardized record dicts.
        """
        ...


# ─────────────── yfinance News ───────────────

class YFinanceNewsFetcher(NewsFetcher):
    """Fetch news via yfinance .news attribute (free, no key).

    Provides ~8-15 recent headlines per ticker.
    No date range filtering; always returns the latest news.
    """

    def fetch_news(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: int = 100,
    ) -> List[Dict]:
        import yfinance as yf

        ticker = yf.Ticker(query)
        try:
            raw_news = ticker.news or []
        except Exception as e:
            logger.warning(f"yfinance news failed for {query}: {e}")
            return []

        records = []
        for item in raw_news[:max_results]:
            try:
                pub_time = datetime.fromtimestamp(
                    item.get("providerPublishTime", 0)
                )
                records.append(
                    _make_record(
                        title=item.get("title", ""),
                        source=f"yfinance/{item.get('publisher', 'unknown')}",
                        published=pub_time,
                        url=item.get("link", ""),
                        ticker=query,
                    )
                )
            except Exception:
                continue

        logger.info(f"yfinance: {len(records)} headlines for {query}")
        return records


# ─────────────── NewsAPI ───────────────

class NewsAPIFetcher(NewsFetcher):
    """Fetch news from newsapi.org.

    Free tier: 100 requests/day, 1-month history.
    Get a key at https://newsapi.org/register

    Parameters
    ----------
    api_key : str
        NewsAPI key. Can also be set via NEWSAPI_KEY env var.
    """

    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.environ.get("NEWSAPI_KEY", "")
        if not self.api_key:
            logger.warning(
                "No NewsAPI key provided. Set NEWSAPI_KEY env var or pass api_key."
            )

    def fetch_news(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: int = 100,
    ) -> List[Dict]:
        if not self.api_key:
            logger.warning("NewsAPI key not set, skipping.")
            return []

        try:
            from newsapi import NewsApiClient
        except ImportError:
            logger.warning("newsapi-python not installed. pip install newsapi-python")
            return []

        client = NewsApiClient(api_key=self.api_key)

        # Default date range: last 7 days (free tier limit)
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        try:
            response = client.get_everything(
                q=query,
                from_param=start_date,
                to=end_date,
                language="en",
                sort_by="publishedAt",
                page_size=min(max_results, 100),
            )
        except Exception as e:
            logger.error(f"NewsAPI request failed: {e}")
            return []

        articles = response.get("articles", [])
        records = []
        for article in articles:
            try:
                pub = datetime.fromisoformat(
                    article["publishedAt"].replace("Z", "+00:00")
                )
                records.append(
                    _make_record(
                        title=article.get("title", ""),
                        source=f"newsapi/{article.get('source', {}).get('name', 'unknown')}",
                        published=pub,
                        url=article.get("url", ""),
                        ticker=query,
                        description=article.get("description", ""),
                    )
                )
            except Exception:
                continue

        logger.info(f"NewsAPI: {len(records)} articles for '{query}'")
        return records


# ─────────────── GDELT (Free / No Key) ───────────────

class GDELTNewsFetcher(NewsFetcher):
    """Fetch news from the GDELT DOC API (free, unlimited, no key).

    GDELT indexes global news every 15 minutes with 3-month history.
    """

    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def fetch_news(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: int = 100,
    ) -> List[Dict]:
        import requests

        params = {
            "query": f'"{query}" sourcelang:eng',
            "mode": "ArtList",
            "maxrecords": str(min(max_results, 250)),
            "format": "json",
            "sort": "DateDesc",
        }

        if start_date:
            params["startdatetime"] = start_date.replace("-", "") + "000000"
        if end_date:
            params["enddatetime"] = end_date.replace("-", "") + "235959"

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"GDELT request failed: {e}")
            return []

        articles = data.get("articles", [])
        records = []
        for article in articles:
            try:
                pub = datetime.strptime(
                    article.get("seendate", "")[:14], "%Y%m%d%H%M%S"
                )
                records.append(
                    _make_record(
                        title=article.get("title", ""),
                        source=f"gdelt/{article.get('domain', 'unknown')}",
                        published=pub,
                        url=article.get("url", ""),
                        ticker=query,
                    )
                )
            except Exception:
                continue

        logger.info(f"GDELT: {len(records)} articles for '{query}'")
        return records


# ─────────────── Multi-Source Aggregator ───────────────

class AggregateNewsFetcher:
    """Combine multiple news sources and deduplicate.

    Parameters
    ----------
    newsapi_key : str, optional
        NewsAPI key. Pass None to skip NewsAPI.
    use_gdelt : bool
        Whether to include GDELT (recommended — free, no key).
    """

    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        use_gdelt: bool = True,
    ):
        self.sources: List[NewsFetcher] = [YFinanceNewsFetcher()]

        if newsapi_key:
            self.sources.append(NewsAPIFetcher(api_key=newsapi_key))

        if use_gdelt:
            self.sources.append(GDELTNewsFetcher())

    def fetch_all(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_per_source: int = 100,
    ) -> pd.DataFrame:
        """Fetch from all sources, deduplicate, return sorted DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: title, source, published, url, ticker, description
            Sorted by published date descending.
        """
        all_records = []
        for source in self.sources:
            try:
                records = source.fetch_news(
                    query, start_date, end_date, max_per_source
                )
                all_records.extend(records)
            except Exception as e:
                logger.warning(f"Source {type(source).__name__} failed: {e}")

        if not all_records:
            logger.warning(f"No news found for '{query}'")
            return pd.DataFrame(
                columns=["title", "source", "published", "url", "ticker", "description"]
            )

        df = pd.DataFrame(all_records)

        # Deduplicate by title similarity (exact match)
        df = df.drop_duplicates(subset=["title"], keep="first")

        # Sort by date
        df = df.sort_values("published", ascending=False).reset_index(drop=True)

        logger.info(
            f"Aggregated {len(df)} unique articles for '{query}' "
            f"from {len(self.sources)} sources"
        )
        return df
