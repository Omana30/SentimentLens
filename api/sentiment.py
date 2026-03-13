"""
api/sentiment.py

Sentiment analysis orchestration layer.

Ties together the NewsAPIClient, SentimentScorer, and FinancialLexicon into a
single SentimentAnalyser class that produces fully-enriched analysis objects
ready for the FastAPI endpoints and the Dash dashboard to consume.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SentimentAnalyser:
    """
    Orchestrates end-to-end financial sentiment analysis for a ticker.

    On initialisation, creates instances of:
        - SentimentScorer   (FinBERT pipeline)
        - FinancialLexicon  (domain lexicon scorer)
        - NewsAPIClient     (news fetcher)

    All three are reused across multiple calls to avoid repeated model
    loading and object creation overhead.
    """

    def __init__(self) -> None:
        """Initialise all sub-components."""
        # Import here to allow tests to patch at the module level
        from models.scorer import SentimentScorer
        from models.lexicon import FinancialLexicon
        from api.news import NewsAPIClient

        self.scorer = SentimentScorer()
        self.lexicon = FinancialLexicon()
        self.news_client = NewsAPIClient(api_key=os.getenv("NEWSAPI_KEY", ""))

    # ------------------------------------------------------------------
    # Article-level analysis
    # ------------------------------------------------------------------

    def analyse_article(self, article: dict) -> dict:
        """
        Score a single article dict and return it with sentiment fields added.

        Combines the article's title and description into one text, scores it
        with FinBERT via SentimentScorer, then adjusts the score using the
        FinancialLexicon.

        Args:
            article: Dict with at minimum "title" and "description" keys.

        Returns:
            Copy of ``article`` with additional keys:
                - combined_text (str):       Title + description.
                - sentiment (str):           Adjusted sentiment label.
                - confidence (float):        Adjusted confidence.
                - lexicon_influence (float): Lexicon score that drove adjustment.
        """
        title: str = article.get("title") or ""
        description: str = article.get("description") or ""
        combined_text: str = f"{title}. {description}".strip()

        # Step 1: FinBERT scoring
        base_score = self.scorer.score_text(combined_text)

        # Step 2: Lexicon adjustment
        adjusted = self.lexicon.adjust_score(
            text=combined_text,
            base_confidence=base_score["confidence"],
            base_sentiment=base_score["sentiment"],
        )

        return {
            **article,
            "combined_text": combined_text,
            "sentiment": adjusted["sentiment"],
            "confidence": adjusted["confidence"],
            "lexicon_influence": adjusted["lexicon_influence"],
        }

    # ------------------------------------------------------------------
    # Ticker-level analysis
    # ------------------------------------------------------------------

    def analyse_ticker(self, ticker: str, company_name: str) -> dict:
        """
        Fetch news for a ticker, analyse all articles, and aggregate results.

        Builds a daily sentiment trend by grouping articles by publication
        date and computing the mean sentiment score for each day (positive=1,
        neutral=0, negative=-1 for trend purposes).

        Args:
            ticker:       Stock symbol, e.g. "AAPL".
            company_name: Full company name, e.g. "Apple".

        Returns:
            dict with keys:
                - ticker (str)
                - company_name (str)
                - articles (list[dict]): Fully-annotated article dicts.
                - daily_trend (list[dict]): [{date, avg_sentiment_score}, …]
                - aggregate (dict): From SentimentScorer.aggregate_scores.
                - analysed_at (str): ISO 8601 timestamp of analysis run.
        """
        # Fetch news (falls back to mock data on failure)
        raw_articles = self.news_client.get_headlines(ticker, company_name)

        # Analyse each article
        analysed_articles: list[dict] = []
        for article in raw_articles:
            try:
                analysed_articles.append(self.analyse_article(article))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to analyse article '%s': %s", article.get("title"), exc)

        # Build daily trend: group by date, compute mean numeric sentiment
        daily_buckets: dict[str, list[float]] = defaultdict(list)
        sentiment_to_score = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

        for article in analysed_articles:
            pub = article.get("published_at") or ""
            date_str = pub[:10] if len(pub) >= 10 else "unknown"
            score = sentiment_to_score.get(article.get("sentiment", "neutral"), 0.0)
            daily_buckets[date_str].append(score)

        daily_trend = [
            {
                "date": date,
                "avg_sentiment_score": round(
                    sum(scores) / len(scores), 4
                ),
            }
            for date, scores in sorted(daily_buckets.items())
        ]

        # Aggregate all scores into overall signal
        score_dicts = [
            {
                "sentiment": a.get("sentiment", "neutral"),
                "confidence": a.get("confidence", 0.5),
            }
            for a in analysed_articles
        ]
        aggregate = self.scorer.aggregate_scores(score_dicts)

        return {
            "ticker": ticker,
            "company_name": company_name,
            "articles": analysed_articles,
            "daily_trend": daily_trend,
            "aggregate": aggregate,
            "analysed_at": datetime.now(tz=timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Price data
    # ------------------------------------------------------------------

    def get_price_data(self, ticker: str, days_back: int = 7) -> dict:
        """
        Retrieve recent OHLCV price data for a ticker using yfinance.

        Args:
            ticker:    Stock symbol, e.g. "AAPL".
            days_back: Number of calendar days of history to fetch.

        Returns:
            dict with keys:
                - dates (list[str]):   ISO date strings.
                - prices (list[float]): Closing prices.
                - change_pct (float):  Overall % change across the period.
        """
        try:
            import yfinance as yf  # type: ignore

            period_map = {7: "7d", 14: "14d", 30: "1mo"}
            period = period_map.get(days_back, f"{days_back}d")

            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period=period, interval="1d")

            if hist.empty:
                raise ValueError(f"yfinance returned empty data for {ticker}")

            dates = [str(d.date()) for d in hist.index]
            prices = [round(float(p), 2) for p in hist["Close"].tolist()]

            # Overall % change: first close → last close
            if len(prices) >= 2 and prices[0] != 0:
                change_pct = round(((prices[-1] - prices[0]) / prices[0]) * 100, 2)
            else:
                change_pct = 0.0

            return {
                "dates": dates,
                "prices": prices,
                "change_pct": change_pct,
            }

        except Exception as exc:  # noqa: BLE001
            logger.warning("get_price_data failed for %s: %s", ticker, exc)
            # Return empty structure rather than crashing
            return {
                "dates": [],
                "prices": [],
                "change_pct": 0.0,
            }
