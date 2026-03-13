"""
api/main.py

FastAPI application entry point for SentimentLens.

Exposes REST endpoints consumed by the Plotly Dash dashboard.  All API keys
are loaded from environment variables via python-dotenv — never hardcoded.

Run with:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv  # type: ignore
from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore

# Load .env before anything else so NEWSAPI_KEY is available to sub-modules
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to tickers data file relative to project root
TICKERS_PATH = Path(__file__).parent.parent / "data" / "tickers.json"

# ------------------------------------------------------------------
# App initialisation
# ------------------------------------------------------------------

app = FastAPI(
    title="SentimentLens API",
    description="Real-time AI-powered financial sentiment analysis",
    version="1.0.0",
)

# Allow the Dash frontend (default port 8050) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load the analyser once on first use (avoids loading FinBERT at import)
_analyser = None


def _get_analyser():
    """Return a singleton SentimentAnalyser instance, creating it on first call."""
    global _analyser  # noqa: PLW0603
    if _analyser is None:
        from api.sentiment import SentimentAnalyser  # noqa: PLC0415

        _analyser = SentimentAnalyser()
    return _analyser


# ------------------------------------------------------------------
# Startup event
# ------------------------------------------------------------------


@app.on_event("startup")
async def startup_event() -> None:
    """Log a startup message when uvicorn is ready."""
    print("SentimentLens API running — http://localhost:8000")
    logger.info("SentimentLens API started successfully.")


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@app.get("/health", summary="Health check")
async def health() -> dict:
    """
    Health check endpoint.

    Returns:
        JSON with status "ok" and current UTC timestamp.
    """
    return {
        "status": "ok",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }


@app.get("/tickers", summary="List supported tickers")
async def get_tickers() -> list[dict]:
    """
    Return the full list of supported tickers from data/tickers.json.

    Returns:
        List of ticker objects with symbol, company_name, and sector.
    """
    try:
        with open(TICKERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read tickers.json: %s", exc)
        raise HTTPException(status_code=500, detail="Could not load tickers data.") from exc


@app.get("/analyse/{ticker}", summary="Analyse sentiment for a ticker")
async def analyse_ticker(ticker: str) -> dict:
    """
    Run full sentiment analysis for a ticker symbol.

    Fetches recent news, scores each article with FinBERT + lexicon
    enhancement, builds a 7-day trend, and returns an aggregate signal.

    Args:
        ticker: Stock symbol (case-insensitive), e.g. "AAPL".

    Returns:
        Full analysis dict from SentimentAnalyser.analyse_ticker.
    """
    ticker = ticker.upper()

    try:
        # Look up company name from tickers.json
        with open(TICKERS_PATH, "r", encoding="utf-8") as f:
            tickers: list[dict] = json.load(f)

        ticker_info = next((t for t in tickers if t["symbol"] == ticker), None)
        company_name = ticker_info["company_name"] if ticker_info else ticker

        analyser = _get_analyser()
        result = analyser.analyse_ticker(ticker=ticker, company_name=company_name)
        return result

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Analysis failed for %s: %s", ticker, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment analysis failed for {ticker}: {exc}",
        ) from exc


@app.get("/price/{ticker}", summary="Get stock price history")
async def get_price(ticker: str) -> dict:
    """
    Return 7-day closing price history for a ticker via yfinance.

    Args:
        ticker: Stock symbol (case-insensitive), e.g. "AAPL".

    Returns:
        dict with dates, prices, and change_pct.
    """
    ticker = ticker.upper()

    try:
        analyser = _get_analyser()
        return analyser.get_price_data(ticker=ticker, days_back=7)
    except Exception as exc:  # noqa: BLE001
        logger.error("Price fetch failed for %s: %s", ticker, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Price data fetch failed for {ticker}: {exc}",
        ) from exc
