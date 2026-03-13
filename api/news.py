"""
api/news.py

NewsAPI client for fetching financial headlines.

Attempts to query the live NewsAPI everything endpoint first.  If the API
is unavailable, rate-limited, or misconfigured, falls back to realistic
mock articles so the rest of the pipeline can continue to function without
interruption.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


class NewsAPIClient:
    """
    Thin wrapper around the NewsAPI 'everything' endpoint.

    Handles authentication, query construction, response parsing, and
    fallback to mock data when the live API is unavailable.

    Args:
        api_key: NewsAPI key.  Defaults to the NEWSAPI_KEY env variable.
    """

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str | None = None) -> None:
        """Initialise the client with an API key from the argument or env."""
        self.api_key: str = api_key or os.getenv("NEWSAPI_KEY", "")
        if not self.api_key:
            logger.warning(
                "NEWSAPI_KEY not set. NewsAPIClient will fall back to mock data."
            )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_headlines(
        self,
        ticker: str,
        company_name: str,
        days_back: int = 7,
    ) -> list[dict]:
        """
        Fetch recent news articles about a ticker / company from NewsAPI.

        Queries the 'everything' endpoint with a compound query of the form
        "{ticker} OR {company_name} stock" over the past ``days_back`` days.

        Args:
            ticker:       Stock symbol, e.g. "AAPL".
            company_name: Full company name, e.g. "Apple".
            days_back:    How many days of history to retrieve (default 7).

        Returns:
            List of raw article dicts as returned by NewsAPI, parsed via
            parse_articles.  Returns mock data on any error.
        """
        if not self.api_key:
            logger.info("No API key — returning mock data for %s.", ticker)
            return self.get_mock_data(ticker)

        try:
            import requests  # type: ignore

            from_date = (
                datetime.now(tz=timezone.utc) - timedelta(days=days_back)
            ).strftime("%Y-%m-%d")

            params: dict[str, Any] = {
                "q": f"{ticker} OR {company_name} stock",
                "from": from_date,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": 20,
                "apiKey": self.api_key,
            }

            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()

            data: dict = response.json()
            raw_articles: list = data.get("articles", [])
            logger.info(
                "NewsAPI returned %d articles for %s.", len(raw_articles), ticker
            )
            return self.parse_articles(raw_articles)

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "NewsAPI request failed for %s: %s. Using mock data.", ticker, exc
            )
            return self.get_mock_data(ticker)

    def parse_articles(self, articles: list) -> list[dict]:
        """
        Clean and normalise a list of raw NewsAPI article dicts.

        Filters out articles missing a title or description.  Extracts only
        the fields needed downstream to keep memory usage low.

        Args:
            articles: Raw article list from NewsAPI JSON response.

        Returns:
            List of normalised article dicts, each containing:
                - title (str)
                - description (str)
                - published_at (str): ISO 8601 timestamp string.
                - url (str)
                - source (str): Source publication name.
        """
        parsed: list[dict] = []

        for article in articles:
            if not isinstance(article, dict):
                continue

            title: str = (article.get("title") or "").strip()
            description: str = (article.get("description") or "").strip()

            # Skip articles with missing essential text fields
            if not title or not description:
                continue

            # Skip NewsAPI's "[Removed]" placeholder articles
            if title == "[Removed]":
                continue

            source_obj = article.get("source") or {}
            source_name: str = (
                source_obj.get("name") or "Unknown"
                if isinstance(source_obj, dict)
                else "Unknown"
            )

            parsed.append(
                {
                    "title": title,
                    "description": description,
                    "published_at": article.get("publishedAt") or "",
                    "url": article.get("url") or "",
                    "source": source_name,
                }
            )

        return parsed

    def get_mock_data(self, ticker: str) -> list[dict]:
        """
        Return 5 realistic mock articles for any ticker.

        Used as a fallback when the live NewsAPI is unavailable or
        rate-limited.  Articles mimic the structure and tone of real
        financial journalism.

        Args:
            ticker: Stock symbol (used to personalise headlines).

        Returns:
            List of 5 article dicts in the same format as parse_articles.
        """
        today = datetime.now(tz=timezone.utc)

        def _iso(days_ago: int) -> str:
            return (today - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")

        return [
            {
                "title": f"{ticker} Shares Rise on Strong Quarterly Earnings Beat",
                "description": (
                    f"{ticker} reported earnings per share well above analyst "
                    "estimates, driven by robust demand and margin expansion. "
                    "Investors responded positively, pushing shares higher in "
                    "after-hours trading."
                ),
                "published_at": _iso(0),
                "url": f"https://example.com/{ticker.lower()}-earnings",
                "source": "Reuters",
            },
            {
                "title": f"Analysts Raise Price Targets for {ticker} After Guidance Upgrade",
                "description": (
                    f"Several Wall Street firms raised their price targets on {ticker} "
                    "following management's upward revision of full-year guidance. "
                    "The consensus now points to continued revenue growth through "
                    "the remainder of the fiscal year."
                ),
                "published_at": _iso(1),
                "url": f"https://example.com/{ticker.lower()}-targets",
                "source": "Bloomberg",
            },
            {
                "title": f"{ticker} Announces Strategic Partnership to Expand Market Reach",
                "description": (
                    f"{ticker} entered into a multi-year strategic partnership "
                    "agreement aimed at accelerating international expansion. "
                    "The deal is expected to add meaningful revenue over the next "
                    "two to three fiscal years."
                ),
                "published_at": _iso(2),
                "url": f"https://example.com/{ticker.lower()}-partnership",
                "source": "Financial Times",
            },
            {
                "title": f"{ticker} Faces Regulatory Scrutiny Over Data Practices",
                "description": (
                    f"Regulators have opened a preliminary inquiry into {ticker}'s "
                    "data handling procedures. The company stated it is fully "
                    "cooperating and believes its practices comply with all applicable "
                    "laws and regulations."
                ),
                "published_at": _iso(4),
                "url": f"https://example.com/{ticker.lower()}-regulatory",
                "source": "Wall Street Journal",
            },
            {
                "title": f"{ticker} Reports Mixed Results Amid Macroeconomic Headwinds",
                "description": (
                    f"{ticker} delivered revenue in line with estimates but earnings "
                    "fell short as rising input costs compressed margins. Management "
                    "cited ongoing supply chain pressures and cautious consumer "
                    "spending as key headwinds in the near term."
                ),
                "published_at": _iso(6),
                "url": f"https://example.com/{ticker.lower()}-results",
                "source": "CNBC",
            },
        ]
