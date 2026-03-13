"""
tests/test_sentiment.py

Unit tests for models/scorer.py and models/lexicon.py.

HuggingFace pipeline is mocked — no real model is loaded during tests.
All assertions verify the shape and semantics of return values rather than
specific model predictions.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Stub out `transformers` so tests never attempt to download FinBERT
# ---------------------------------------------------------------------------

def _make_mock_pipeline(*args, **kwargs):
    """Return a callable that mimics the HuggingFace pipeline."""
    def _pipeline(text):
        return [{"label": "positive", "score": 0.87}]
    return _pipeline


def _patch_transformers():
    """Insert a minimal transformers stub into sys.modules."""
    transformers_mock = types.ModuleType("transformers")
    transformers_mock.pipeline = _make_mock_pipeline
    sys.modules.setdefault("transformers", transformers_mock)


_patch_transformers()


# ---------------------------------------------------------------------------
# Now import the real modules under test
# ---------------------------------------------------------------------------

from models.scorer import SentimentScorer  # noqa: E402
from models.lexicon import FinancialLexicon  # noqa: E402


# ---------------------------------------------------------------------------
# SentimentScorer tests
# ---------------------------------------------------------------------------

class TestSentimentScorerInit(unittest.TestCase):
    """SentimentScorer initialises correctly with a mocked pipeline."""

    def test_initialises_without_error(self):
        scorer = SentimentScorer()
        self.assertIsNotNone(scorer)


class TestScoreText(unittest.TestCase):
    """SentimentScorer.score_text returns well-formed dicts."""

    def setUp(self):
        self.scorer = SentimentScorer()

    def test_returns_dict_with_correct_keys(self):
        result = self.scorer.score_text("Apple reports record quarterly revenue.")
        self.assertIn("sentiment", result)
        self.assertIn("confidence", result)
        self.assertIn("raw_label", result)

    def test_confidence_in_range(self):
        result = self.scorer.score_text("Markets rallied on strong earnings data.")
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_sentiment_valid_label(self):
        result = self.scorer.score_text("Company faces bankruptcy proceedings.")
        self.assertIn(result["sentiment"], {"positive", "negative", "neutral"})

    def test_empty_string_returns_neutral(self):
        result = self.scorer.score_text("")
        self.assertEqual(result["sentiment"], "neutral")
        self.assertEqual(result["confidence"], 0.5)

    def test_none_returns_neutral(self):
        result = self.scorer.score_text(None)  # type: ignore[arg-type]
        self.assertEqual(result["sentiment"], "neutral")

    def test_long_text_does_not_crash(self):
        long_text = "Tesla shares " * 200
        result = self.scorer.score_text(long_text)
        self.assertIn("sentiment", result)


class TestScoreBatch(unittest.TestCase):
    """SentimentScorer.score_batch processes lists correctly."""

    def setUp(self):
        self.scorer = SentimentScorer()

    def test_returns_list_same_length(self):
        texts = ["Apple up 3%", "Microsoft misses guidance", "Neutral market day"]
        results = self.scorer.score_batch(texts)
        self.assertEqual(len(results), 3)

    def test_each_result_has_required_keys(self):
        results = self.scorer.score_batch(["Strong revenue beat"])
        self.assertIn("sentiment", results[0])
        self.assertIn("confidence", results[0])
        self.assertIn("raw_label", results[0])


class TestAggregateScores(unittest.TestCase):
    """SentimentScorer.aggregate_scores produces correct aggregate signals."""

    def setUp(self):
        self.scorer = SentimentScorer()

    def _make_scores(self, sentiment: str, n: int = 5) -> list[dict]:
        return [{"sentiment": sentiment, "confidence": 0.8} for _ in range(n)]

    def test_returns_dict_with_correct_keys(self):
        result = self.scorer.aggregate_scores(self._make_scores("positive"))
        for key in ("signal", "confidence", "positive_pct", "negative_pct", "neutral_pct", "article_count"):
            self.assertIn(key, result)

    def test_all_positive_returns_bullish(self):
        result = self.scorer.aggregate_scores(self._make_scores("positive"))
        self.assertEqual(result["signal"], "Bullish")

    def test_all_negative_returns_bearish(self):
        result = self.scorer.aggregate_scores(self._make_scores("negative"))
        self.assertEqual(result["signal"], "Bearish")

    def test_all_neutral_returns_neutral(self):
        result = self.scorer.aggregate_scores(self._make_scores("neutral"))
        self.assertEqual(result["signal"], "Neutral")

    def test_empty_list_returns_neutral(self):
        result = self.scorer.aggregate_scores([])
        self.assertEqual(result["signal"], "Neutral")
        self.assertEqual(result["article_count"], 0)

    def test_article_count_correct(self):
        scores = self._make_scores("positive", n=7)
        result = self.scorer.aggregate_scores(scores)
        self.assertEqual(result["article_count"], 7)

    def test_positive_pct_sums_correctly(self):
        # 4 positive, 1 negative → positive_pct should be 0.8
        scores = self._make_scores("positive", 4) + self._make_scores("negative", 1)
        result = self.scorer.aggregate_scores(scores)
        self.assertAlmostEqual(result["positive_pct"], 0.8, places=2)

    def test_confidence_is_float_in_range(self):
        result = self.scorer.aggregate_scores(self._make_scores("positive"))
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)


# ---------------------------------------------------------------------------
# FinancialLexicon tests
# ---------------------------------------------------------------------------

class TestFinancialLexiconTermCount(unittest.TestCase):
    """Lexicon must contain at least 50 terms."""

    def test_at_least_50_terms(self):
        lexicon = FinancialLexicon()
        self.assertGreaterEqual(len(lexicon.LEXICON), 50)


class TestGetLexiconScore(unittest.TestCase):
    """FinancialLexicon.get_lexicon_score returns values in [-1.0, 1.0]."""

    def setUp(self):
        self.lexicon = FinancialLexicon()

    def test_returns_float(self):
        score = self.lexicon.get_lexicon_score("Company files for bankruptcy")
        self.assertIsInstance(score, float)

    def test_score_in_range(self):
        score = self.lexicon.get_lexicon_score("Record revenue and strong guidance raised")
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)

    def test_no_financial_terms_returns_zero(self):
        score = self.lexicon.get_lexicon_score("The weather is nice today.")
        self.assertEqual(score, 0.0)

    def test_empty_string_returns_zero(self):
        score = self.lexicon.get_lexicon_score("")
        self.assertEqual(score, 0.0)

    def test_positive_term_gives_positive_score(self):
        score = self.lexicon.get_lexicon_score("Company reports record revenue this quarter.")
        self.assertGreater(score, 0.0)

    def test_negative_term_gives_negative_score(self):
        score = self.lexicon.get_lexicon_score("Company files for bankruptcy protection.")
        self.assertLess(score, 0.0)


class TestAdjustScore(unittest.TestCase):
    """FinancialLexicon.adjust_score returns correctly structured dicts."""

    def setUp(self):
        self.lexicon = FinancialLexicon()

    def test_returns_dict_with_correct_keys(self):
        result = self.lexicon.adjust_score("Record revenue beat", 0.75, "positive")
        self.assertIn("sentiment", result)
        self.assertIn("confidence", result)
        self.assertIn("lexicon_influence", result)

    def test_confidence_stays_in_range(self):
        result = self.lexicon.adjust_score("Beat expectations and upgrade", 0.9, "positive")
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_lexicon_influence_is_float(self):
        result = self.lexicon.adjust_score("Bankruptcy and fraud scandal", 0.6, "neutral")
        self.assertIsInstance(result["lexicon_influence"], float)

    def test_strong_negative_lexicon_overrides_positive_finbert(self):
        # FinBERT says positive but strong negative lexicon signal should override
        result = self.lexicon.adjust_score(
            "Company files for bankruptcy and fraud charges",
            0.55,
            "positive",
        )
        self.assertEqual(result["sentiment"], "negative")


if __name__ == "__main__":
    unittest.main()
