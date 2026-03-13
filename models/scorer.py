"""
scorer.py — Lightweight financial sentiment scorer using VADER + lexicon enhancement.

Uses VADER sentiment analysis optimised for financial text — no heavy ML dependencies,
runs on free hosting. FinBERT can be swapped in for local/production use.

Part of SentimentLens by Omana Prabhakar (github.com/Omana30)
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SentimentScorer:
    """
    Financial sentiment scorer using VADER with financial domain tuning.

    Designed to run on free hosting tiers without heavy ML dependencies.
    For higher accuracy, swap the _score_with_vader method with FinBERT
    when running locally with sufficient RAM.
    """

    def __init__(self):
        """Initialise VADER analyser. Loads once and caches as instance variable."""
        self.analyser = SentimentIntensityAnalyzer()
        self._add_financial_terms()
        logger.info("SentimentScorer initialised with VADER + financial lexicon")

    def _add_financial_terms(self):
        """Add financial domain terms to VADER lexicon for better accuracy."""
        financial_additions = {
            "bullish": 2.0, "bearish": -2.0,
            "outperform": 1.5, "underperform": -1.5,
            "beat": 1.3, "miss": -1.3,
            "surge": 2.0, "plunge": -2.0,
            "rally": 1.8, "crash": -2.5,
            "upgrade": 1.5, "downgrade": -1.5,
            "record": 1.2, "bankruptcy": -3.0,
            "fraud": -2.8, "lawsuit": -1.8,
            "dividend": 1.2, "buyback": 1.3,
            "acquisition": 1.0, "layoffs": -1.5,
        }
        self.analyser.lexicon.update(financial_additions)

    def score_text(self, text: str) -> dict:
        """
        Score a single text string for financial sentiment.

        Args:
            text: Text to analyse. Truncated to 512 chars.

        Returns:
            dict with keys: sentiment (str), confidence (float), raw_label (str)
        """
        try:
            truncated = text[:512] if len(text) > 512 else text
            scores = self.analyser.polarity_scores(truncated)
            compound = scores["compound"]

            if compound >= 0.05:
                sentiment = "positive"
                confidence = min(0.5 + compound * 0.5, 1.0)
            elif compound <= -0.05:
                sentiment = "negative"
                confidence = min(0.5 + abs(compound) * 0.5, 1.0)
            else:
                sentiment = "neutral"
                confidence = 1.0 - abs(compound) * 2

            return {
                "sentiment": sentiment,
                "confidence": round(confidence, 3),
                "raw_label": f"VADER:{sentiment.upper()}",
                "compound_score": round(compound, 3)
            }

        except Exception as e:
            logger.warning(f"Scoring failed: {e}. Returning neutral.")
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "raw_label": "FALLBACK:NEUTRAL",
                "compound_score": 0.0
            }

    def score_batch(self, texts: list[str]) -> list[dict]:
        """
        Score a list of texts.

        Args:
            texts: List of strings to score.

        Returns:
            List of score dicts matching order of input texts.
        """
        return [self.score_text(text) for text in texts]

    def aggregate_scores(self, scores: list[dict]) -> dict:
        """
        Aggregate multiple article scores into an overall market signal.

        Args:
            scores: List of score dicts from score_text or score_batch.

        Returns:
            dict with: signal, confidence, positive_pct, negative_pct,
                      neutral_pct, article_count
        """
        if not scores:
            return {
                "signal": "Neutral",
                "confidence": 0.0,
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "neutral_pct": 0.0,
                "article_count": 0
            }

        total = len(scores)
        positive = sum(1 for s in scores if s["sentiment"] == "positive")
        negative = sum(1 for s in scores if s["sentiment"] == "negative")
        neutral = total - positive - negative

        positive_pct = round(positive / total * 100, 1)
        negative_pct = round(negative / total * 100, 1)
        neutral_pct = round(neutral / total * 100, 1)

        avg_confidence = round(
            sum(s["confidence"] for s in scores) / total, 3
        )

        if positive_pct > 50:
            signal = "Bullish"
        elif negative_pct > 50:
            signal = "Bearish"
        else:
            signal = "Neutral"

        return {
            "signal": signal,
            "confidence": avg_confidence,
            "positive_pct": positive_pct,
            "negative_pct": negative_pct,
            "neutral_pct": neutral_pct,
            "article_count": total
        }