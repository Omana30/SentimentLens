"""
tests/test_api.py

Unit tests for api/news.py and api/sentiment.py.

All external calls (NewsAPI HTTP requests, yfinance, HuggingFace pipeline)
are mocked via unittest.mock so tests run offline without any API keys.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Stub transformers (prevents FinBERT download)
# ---------------------------------------------------------------------------

def _make_mock_pipeline(*args, **kwargs):
    def _pipeline(text):
        return [{"label": "positive", "score": 0.87}]
    return _pipeline


transformers_mock = types.ModuleType("transformers")
transformers_mock.pipeline = _make_mock_pipeline
sys.modules.setdefault("transformers", transformers_mock)


# ---------------------------------------------------------------------------
# Stub yfinance
# ---------------------------------------------------------------------------

import pandas as pd  # noqa: E402 — needed for mock DataFrame

_mock_hist = pd.DataFrame(
    {"Close": [150.0, 152.0, 148.0, 155.0, 157.0, 160.0, 158.0]},
    index=pd.date_range("2024-01-01", periods=7, freq="D"),
)

yf_mock = types.ModuleType("yfinance")
_ticker_mock = MagicMock()
_ticker_mock.history.return_value = _mock_hist
yf_mock.Ticker = MagicMock(return_value=_ticker_mock)
sys.modules.setdefault("yfinance", yf_mock)


# ---------------------------------------------------------------------------
# Import modules under test
# ---------------------------------------------------------------------------

from api.news import NewsAPIClient  # noqa: E402
from api.sentiment import SentimentAnalyser  # noqa: E402


# ---------------------------------------------------------------------------
# NewsAPIClient tests
# ---------------------------------------------------------------------------

class TestNewsAPIClientParseArticles(unittest.TestCase):
    """parse_articles filters and normalises raw NewsAPI article dicts."""

    def setUp(self):
        self.client = NewsAPIClient(api_key="fake-key")

    def _raw_article(self, title="Apple beats earnings", description="Record quarter."):
        return {
            "title": title,
            "description": description,
            "publishedAt": "2024-01-15T10:00:00Z",
            "url": "https://example.com/article",
            "source": {"name": "Reuters"},
        }

    def test_returns_list(self):
        result = self.client.parse_articles([self._raw_article()])
        self.assertIsInstance(result, list)

    def test_returns_correct_structure(self):
        result = self.client.parse_articles([self._raw_article()])
        self.assertEqual(len(result), 1)
        article = result[0]
        for key in ("title", "description", "published_at", "url", "source"):
            self.assertIn(key, article)

    def test_filters_articles_with_no_title(self):
        articles = [
            self._raw_article(title=""),   # no title — should be filtered
            self._raw_article(),           # valid
        ]
        result = self.client.parse_articles(articles)
        self.assertEqual(len(result), 1)

    def test_filters_articles_with_no_description(self):
        articles = [
            self._raw_article(description=""),   # no description — filtered
            self._raw_article(),                 # valid
        ]
        result = self.client.parse_articles(articles)
        self.assertEqual(len(result), 1)

    def test_filters_removed_placeholder(self):
        articles = [
            self._raw_article(title="[Removed]"),   # NewsAPI placeholder
            self._raw_article(),
        ]
        result = self.client.parse_articles(articles)
        self.assertEqual(len(result), 1)

    def test_empty_list_returns_empty(self):
        result = self.client.parse_articles([])
        self.assertEqual(result, [])

    def test_source_name_extracted(self):
        result = self.client.parse_articles([self._raw_article()])
        self.assertEqual(result[0]["source"], "Reuters")

    def test_published_at_key_present(self):
        result = self.client.parse_articles([self._raw_article()])
        self.assertIn("published_at", result[0])
        self.assertEqual(result[0]["published_at"], "2024-01-15T10:00:00Z")


class TestNewsAPIClientMockData(unittest.TestCase):
    """get_mock_data always returns 5 well-formed articles."""

    def setUp(self):
        self.client = NewsAPIClient(api_key="")

    def test_returns_5_articles(self):
        articles = self.client.get_mock_data("AAPL")
        self.assertEqual(len(articles), 5)

    def test_articles_have_required_keys(self):
        articles = self.client.get_mock_data("TSLA")
        for article in articles:
            for key in ("title", "description", "published_at", "url", "source"):
                self.assertIn(key, article, msg=f"Key '{key}' missing from mock article")

    def test_works_for_any_ticker(self):
        for ticker in ["AAPL", "MSFT", "NVDA", "BABA"]:
            articles = self.client.get_mock_data(ticker)
            self.assertEqual(len(articles), 5)

    def test_titles_contain_ticker(self):
        articles = self.client.get_mock_data("GOOGL")
        # At least the first article should mention the ticker
        self.assertIn("GOOGL", articles[0]["title"])


# ---------------------------------------------------------------------------
# SentimentAnalyser tests
# ---------------------------------------------------------------------------

class TestSentimentAnalyserAnalyseArticle(unittest.TestCase):
    """analyse_article adds sentiment fields to an article dict."""

    def setUp(self):
        self.analyser = SentimentAnalyser()

    def _sample_article(self):
        return {
            "title": "Apple reports record revenue and strong guidance",
            "description": "Q4 results beat expectations; buyback announced.",
            "published_at": "2024-01-15T10:00:00Z",
            "url": "https://example.com/apple",
            "source": "Reuters",
        }

    def test_returns_dict(self):
        result = self.analyser.analyse_article(self._sample_article())
        self.assertIsInstance(result, dict)

    def test_adds_sentiment_key(self):
        result = self.analyser.analyse_article(self._sample_article())
        self.assertIn("sentiment", result)

    def test_adds_confidence_key(self):
        result = self.analyser.analyse_article(self._sample_article())
        self.assertIn("confidence", result)

    def test_adds_lexicon_influence_key(self):
        result = self.analyser.analyse_article(self._sample_article())
        self.assertIn("lexicon_influence", result)

    def test_adds_combined_text_key(self):
        result = self.analyser.analyse_article(self._sample_article())
        self.assertIn("combined_text", result)

    def test_original_keys_preserved(self):
        article = self._sample_article()
        result = self.analyser.analyse_article(article)
        self.assertEqual(result["title"], article["title"])
        self.assertEqual(result["source"], article["source"])

    def test_sentiment_is_valid_label(self):
        result = self.analyser.analyse_article(self._sample_article())
        self.assertIn(result["sentiment"], {"positive", "negative", "neutral"})

    def test_confidence_in_range(self):
        result = self.analyser.analyse_article(self._sample_article())
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)


class TestSentimentAnalyserGetPriceData(unittest.TestCase):
    """get_price_data returns correctly structured price dict."""

    def setUp(self):
        self.analyser = SentimentAnalyser()

    def test_returns_dict(self):
        result = self.analyser.get_price_data("AAPL")
        self.assertIsInstance(result, dict)

    def test_has_dates_key(self):
        result = self.analyser.get_price_data("AAPL")
        self.assertIn("dates", result)

    def test_has_prices_key(self):
        result = self.analyser.get_price_data("AAPL")
        self.assertIn("prices", result)

    def test_has_change_pct_key(self):
        result = self.analyser.get_price_data("AAPL")
        self.assertIn("change_pct", result)

    def test_dates_and_prices_same_length(self):
        result = self.analyser.get_price_data("AAPL")
        self.assertEqual(len(result["dates"]), len(result["prices"]))

    def test_change_pct_is_float(self):
        result = self.analyser.get_price_data("AAPL")
        self.assertIsInstance(result["change_pct"], float)

    def test_prices_are_numeric(self):
        result = self.analyser.get_price_data("MSFT")
        for price in result["prices"]:
            self.assertIsInstance(price, (int, float))


if __name__ == "__main__":
    unittest.main()
