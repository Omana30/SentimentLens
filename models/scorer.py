"""
models/scorer.py

FinBERT-based sentiment scorer for financial text.

Loads ProsusAI/finbert once on initialisation and exposes methods for
single-text scoring, batch scoring, and aggregate signal calculation.
All methods handle exceptions gracefully and never propagate crashes to callers.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SentimentScorer:
    """
    Wraps the ProsusAI/finbert HuggingFace pipeline for financial sentiment
    classification.

    The model is loaded once at init and cached as an instance variable to
    avoid repeated disk I/O across multiple calls.

    Sentiment labels returned by FinBERT are:
        - "positive"  →  Bullish signal
        - "negative"  →  Bearish signal
        - "neutral"   →  No clear directional signal
    """

    # Maximum characters passed to FinBERT (tokeniser limit is ~512 tokens;
    # 512 chars is a safe proxy that avoids truncation errors).
    MAX_TEXT_LENGTH: int = 512

    def __init__(self) -> None:
        """
        Initialise the SentimentScorer by loading ProsusAI/finbert.

        On first run the model (~500 MB) is downloaded from HuggingFace Hub
        and cached in the default HuggingFace cache directory.  Subsequent
        runs load from the local cache.
        """
        self._pipeline: Any = None
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        """Load the FinBERT text-classification pipeline."""
        try:
            from transformers import pipeline  # type: ignore

            self._pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
            )
            logger.info("FinBERT pipeline loaded successfully.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load FinBERT pipeline: %s", exc)
            self._pipeline = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_text(self, text: str) -> dict:
        """
        Score a single piece of financial text with FinBERT.

        Args:
            text: Raw financial text (headline, description, article body).

        Returns:
            dict with keys:
                - sentiment (str): "positive" | "negative" | "neutral"
                - confidence (float): model confidence in [0.0, 1.0]
                - raw_label (str): label string as returned by the pipeline
        """
        if not text or not isinstance(text, str):
            return self._neutral_score()

        # Truncate to avoid tokeniser overflow
        truncated = text[: self.MAX_TEXT_LENGTH]

        try:
            if self._pipeline is None:
                raise RuntimeError("Pipeline not loaded")

            results = self._pipeline(truncated)
            # Pipeline returns a list; we always pass one text so take [0]
            result = results[0]
            raw_label: str = result.get("label", "neutral")
            confidence: float = float(result.get("score", 0.5))

            # FinBERT labels are already lowercase "positive"/"negative"/"neutral"
            sentiment = raw_label.lower()
            if sentiment not in {"positive", "negative", "neutral"}:
                sentiment = "neutral"

            return {
                "sentiment": sentiment,
                "confidence": round(confidence, 4),
                "raw_label": raw_label,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("score_text failed for text '%s...': %s", text[:50], exc)
            return self._neutral_score()

    def score_batch(self, texts: list[str]) -> list[dict]:
        """
        Score a list of financial texts.

        Args:
            texts: List of text strings to score.

        Returns:
            List of score dicts (same structure as score_text).
            Failed items are replaced with a neutral score rather than
            raising an exception.
        """
        return [self.score_text(t) for t in texts]

    def aggregate_scores(self, scores: list[dict]) -> dict:
        """
        Aggregate a list of per-article scores into an overall market signal.

        Calculates percentage breakdowns across positive / negative / neutral
        articles and maps the dominant sentiment to a directional signal.

        Args:
            scores: List of score dicts as returned by score_text / score_batch.

        Returns:
            dict with keys:
                - signal (str): "Bullish" | "Bearish" | "Neutral"
                - confidence (float): mean confidence across all articles
                - positive_pct (float): fraction of positive articles [0-1]
                - negative_pct (float): fraction of negative articles [0-1]
                - neutral_pct (float):  fraction of neutral articles [0-1]
                - article_count (int):  total number of articles scored
        """
        if not scores:
            return {
                "signal": "Neutral",
                "confidence": 0.0,
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "neutral_pct": 0.0,
                "article_count": 0,
            }

        total = len(scores)
        positive_count = sum(1 for s in scores if s.get("sentiment") == "positive")
        negative_count = sum(1 for s in scores if s.get("sentiment") == "negative")
        neutral_count = total - positive_count - negative_count

        positive_pct = round(positive_count / total, 4)
        negative_pct = round(negative_count / total, 4)
        neutral_pct = round(neutral_count / total, 4)

        # Average confidence across all articles
        mean_confidence = round(
            sum(s.get("confidence", 0.5) for s in scores) / total, 4
        )

        # Determine overall signal from majority class
        if positive_count > negative_count and positive_count > neutral_count:
            signal = "Bullish"
        elif negative_count > positive_count and negative_count > neutral_count:
            signal = "Bearish"
        else:
            signal = "Neutral"

        return {
            "signal": signal,
            "confidence": mean_confidence,
            "positive_pct": positive_pct,
            "negative_pct": negative_pct,
            "neutral_pct": neutral_pct,
            "article_count": total,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _neutral_score() -> dict:
        """Return a safe neutral fallback score."""
        return {
            "sentiment": "neutral",
            "confidence": 0.5,
            "raw_label": "neutral",
        }
