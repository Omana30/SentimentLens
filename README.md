# SentimentLens

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![FinBERT](https://img.shields.io/badge/Model-FinBERT-orange)

> Real-time AI-powered financial sentiment analysis dashboard

SentimentLens analyses live financial news using state-of-the-art NLP to surface market sentiment signals for 10 major tickers. Built as a portfolio project by Omana Prabhakar, AI Product Builder, London.

---

## Built on Research

This project extends MSc thesis research on NLP-driven stock market sentiment prediction conducted at Heriot-Watt University, Edinburgh. The financial lexicon enhancement technique implemented in `models/lexicon.py` directly applies methodology developed during that research. The hybrid approach — combining transformer-based FinBERT classification with domain-specific lexicon scoring — improves sentiment signal precision for financial texts compared to either method alone.

---

## Architecture

```
┌─────────────┐     ┌──────────┐     ┌─────────────────┐
│   NewsAPI   │────▶│ news.py  │────▶│  sentiment.py   │
│  (7-day     │     │ (fetch & │     │  (orchestrate   │
│   window)   │     │  parse)  │     │   analysis)     │
└─────────────┘     └──────────┘     └────────┬────────┘
                                              │
                              ┌───────────────┴───────────────┐
                              │                               │
                    ┌─────────▼────────┐         ┌───────────▼──────────┐
                    │ FinBERT (local)  │         │  Financial Lexicon   │
                    │ ProsusAI/finbert │         │  (50+ domain terms)  │
                    │ HuggingFace Hub  │         │  models/lexicon.py   │
                    └─────────┬────────┘         └───────────┬──────────┘
                              │                               │
                              └───────────────┬───────────────┘
                                              │
                                    ┌─────────▼────────┐
                                    │  FastAPI Backend  │
                                    │  api/main.py      │
                                    │  localhost:8000   │
                                    └─────────┬────────┘
                                              │
                                    ┌─────────▼────────┐
                                    │  Plotly Dash UI  │
                                    │  dashboard/app.py │
                                    │  localhost:8050   │
                                    └──────────────────┘
                                              │
                                    ┌─────────▼────────┐
                                    │    yfinance      │
                                    │  (stock prices,  │
                                    │   free, local)   │
                                    └──────────────────┘
```

---

## Features

- **Real-time sentiment analysis** using FinBERT (ProsusAI/finbert) running locally — no inference costs
- **Hybrid scoring** combining FinBERT predictions with a curated 50+ term financial lexicon
- **7-day sentiment trend charts** showing sentiment trajectory per ticker
- **Live stock price overlays** via yfinance (free, no API key)
- **Bullish / Bearish / Neutral signal aggregation** across all recent articles
- **Dark financial terminal UI** built with Plotly Dash
- **Auto-refresh** every 5 minutes via dcc.Interval
- **Graceful fallback** to mock data when NewsAPI is rate-limited
- **Full test suite** with mocked external dependencies

---

## Tech Stack

| Component | Technology | Cost |
|---|---|---|
| Sentiment Model | ProsusAI/finbert (HuggingFace) | Free (runs locally) |
| News Data | NewsAPI free tier | Free (100 req/day) |
| Stock Prices | yfinance | Free |
| Dashboard | Plotly Dash + dark theme | Free |
| Backend | FastAPI + uvicorn | Free |
| Testing | pytest + unittest.mock | Free |

---

## Setup

### Prerequisites
- Python 3.10+
- A free [NewsAPI key](https://newsapi.org/register)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/SentimentLens.git
cd SentimentLens

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your NEWSAPI_KEY

# 4. Start the FastAPI backend (Terminal 1)
uvicorn api.main:app --reload

# 5. Start the Dash dashboard (Terminal 2)
python dashboard/app.py

# 6. Open in browser
# http://localhost:8050
```

### First run note
On first run, FinBERT (~500MB) will be downloaded from HuggingFace Hub and cached locally. Subsequent runs load from cache instantly.

---

## Running Tests

```bash
pytest tests/ -v
```

Tests use mocked external dependencies — no real API calls, no model loading.

---

## Project Structure

```
SentimentLens/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── dashboard/
│   └── app.py          # Plotly Dash frontend
├── api/
│   ├── __init__.py
│   ├── main.py         # FastAPI endpoints
│   ├── sentiment.py    # Analysis orchestration
│   └── news.py         # NewsAPI client
├── models/
│   ├── __init__.py
│   ├── scorer.py       # FinBERT scorer
│   └── lexicon.py      # Financial term lexicon
├── data/
│   └── tickers.json    # Supported tickers
└── tests/
    ├── __init__.py
    ├── test_sentiment.py
    └── test_api.py
```

---

## License

MIT License — see LICENSE for details.

---

## Author

**Omana Prabhakar** · AI Product Builder · London
Extended from MSc thesis research at Heriot-Watt University, Edinburgh
