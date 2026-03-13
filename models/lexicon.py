"""
models/lexicon.py

Domain-specific financial lexicon for sentiment enhancement.

This module implements the lexicon enhancement technique developed as part of
MSc thesis research on NLP-driven stock market sentiment prediction at
Heriot-Watt University, Edinburgh.  The hybrid approach — combining
transformer-based FinBERT predictions with a curated domain lexicon — was
shown in that research to improve precision on financial texts where
model-agnostic language may otherwise dilute the signal.

The lexicon maps 50+ financial terms to sentiment weights in [-1.0, +1.0].
Negative weights indicate bearish language; positive weights indicate bullish
language.  A score near 0.0 means neutral / no strong signal.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class FinancialLexicon:
    """
    Curated financial-domain lexicon for sentiment score adjustment.

    Scans text for known financial terms and returns a weighted score that can
    shift or reinforce a base FinBERT prediction.  Useful when FinBERT's
    general-language training under-weights specialist financial vocabulary.

    Attributes:
        LEXICON (dict[str, float]): Mapping of term → sentiment weight.
    """

    # ------------------------------------------------------------------
    # Lexicon: 50+ financial terms with sentiment weights in [-1.0, +1.0]
    # Negative = bearish; Positive = bullish; magnitude reflects certainty.
    # ------------------------------------------------------------------
    LEXICON: dict[str, float] = {
        # Strong negative signals
        "bankruptcy": -0.95,
        "bankrupt": -0.95,
        "fraud": -0.90,
        "fraudulent": -0.90,
        "scandal": -0.88,
        "default": -0.85,
        "insolvency": -0.85,
        "insolvent": -0.85,
        "criminal charges": -0.82,
        "sec investigation": -0.80,
        "class action": -0.78,
        "missed expectations": -0.70,
        "missed estimates": -0.70,
        "earnings miss": -0.70,
        "lawsuit": -0.70,
        "litigation": -0.65,
        "downgrade": -0.65,
        "price target cut": -0.65,
        "layoffs": -0.60,
        "job cuts": -0.60,
        "restructuring": -0.55,
        "recall": -0.55,
        "product recall": -0.55,
        "profit warning": -0.75,
        "revenue miss": -0.70,
        "guidance cut": -0.72,
        "margin compression": -0.60,
        "debt downgrade": -0.75,
        "credit downgrade": -0.75,
        "supply chain disruption": -0.58,
        "trade war": -0.55,
        "tariff increase": -0.52,
        "market share loss": -0.60,
        "customer churn": -0.55,
        "write-down": -0.65,
        "write-off": -0.65,
        "impairment": -0.60,
        "fine": -0.50,
        "penalty": -0.52,
        "regulatory action": -0.55,
        # Strong positive signals
        "beat expectations": 0.80,
        "beat estimates": 0.80,
        "earnings beat": 0.80,
        "record revenue": 0.85,
        "record earnings": 0.85,
        "record profit": 0.85,
        "all-time high": 0.82,
        "acquisition": 0.50,
        "merger": 0.48,
        "partnership": 0.55,
        "strategic partnership": 0.60,
        "upgrade": 0.65,
        "price target raised": 0.65,
        "price target increase": 0.65,
        "dividend increase": 0.70,
        "dividend hike": 0.70,
        "buyback": 0.60,
        "share repurchase": 0.60,
        "new contract": 0.60,
        "market share gain": 0.70,
        "market share growth": 0.68,
        "regulatory approval": 0.75,
        "fda approval": 0.80,
        "ipo": 0.50,
        "expansion": 0.55,
        "revenue growth": 0.65,
        "strong guidance": 0.70,
        "guidance raised": 0.72,
        "outperform": 0.65,
        "strong results": 0.68,
        "profitable": 0.58,
        "profitability": 0.55,
        "innovation": 0.45,
        "breakthrough": 0.60,
        "landmark deal": 0.65,
        "strategic investment": 0.55,
        "strong demand": 0.60,
        "subscriber growth": 0.62,
        "user growth": 0.60,
    }

    def get_lexicon_score(self, text: str) -> float:
        """
        Scan text for financial terms and return a weighted average score.

        Performs case-insensitive substring matching against all terms in the
        lexicon.  If multiple terms match, returns their arithmetic mean.

        Args:
            text: Financial text to analyse.

        Returns:
            float in [-1.0, 1.0].  Returns 0.0 if no terms match.
        """
        if not text or not isinstance(text, str):
            return 0.0

        text_lower = text.lower()
        matched_weights: list[float] = []

        for term, weight in self.LEXICON.items():
            if term in text_lower:
                matched_weights.append(weight)

        if not matched_weights:
            return 0.0

        # Weighted average — keep result in [-1.0, 1.0]
        score = sum(matched_weights) / len(matched_weights)
        return round(max(-1.0, min(1.0, score)), 4)

    def adjust_score(
        self,
        text: str,
        base_confidence: float,
        base_sentiment: str,
    ) -> dict:
        """
        Combine a FinBERT base score with the lexicon score to produce an
        adjusted sentiment reading.

        Implementation of the lexicon enhancement technique from MSc thesis
        research.  When the lexicon finds a strong signal (|score| > 0.5),
        the sentiment label is shifted towards the lexicon's direction if it
        conflicts with FinBERT, or confidence is boosted if they agree.

        Args:
            text:             Financial text that was scored.
            base_confidence:  FinBERT confidence in [0.0, 1.0].
            base_sentiment:   FinBERT sentiment label ("positive"/"negative"/"neutral").

        Returns:
            dict with keys:
                - sentiment (str):         Adjusted sentiment label.
                - confidence (float):      Adjusted confidence value.
                - lexicon_influence (float): Lexicon score that drove adjustment.
        """
        lexicon_score = self.get_lexicon_score(text)
        adjusted_sentiment = base_sentiment
        adjusted_confidence = base_confidence

        # Only override / boost when lexicon signal is strong (|score| > 0.5)
        if abs(lexicon_score) > 0.5:
            lexicon_sentiment = "positive" if lexicon_score > 0 else "negative"

            if lexicon_sentiment != base_sentiment:
                # Lexicon disagrees — shift to lexicon direction with moderate confidence
                logger.debug(
                    "Lexicon overrides FinBERT: %s → %s (lexicon=%.3f)",
                    base_sentiment,
                    lexicon_sentiment,
                    lexicon_score,
                )
                adjusted_sentiment = lexicon_sentiment
                # Blend: keep some FinBERT confidence, boost with lexicon signal
                adjusted_confidence = round(
                    (base_confidence * 0.4) + (abs(lexicon_score) * 0.6), 4
                )
            else:
                # Lexicon agrees — boost confidence slightly
                boost = abs(lexicon_score) * 0.1
                adjusted_confidence = round(
                    min(1.0, base_confidence + boost), 4
                )

        return {
            "sentiment": adjusted_sentiment,
            "confidence": adjusted_confidence,
            "lexicon_influence": lexicon_score,
        }
