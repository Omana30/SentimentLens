"""
dashboard/app.py

Plotly Dash dashboard for SentimentLens.

Dark financial terminal aesthetic.  Consumes the FastAPI backend running at
http://localhost:8000 to display real-time sentiment analysis, price charts,
and news feeds for 10 major stock tickers.

Run with:
    python dashboard/app.py
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import requests  # type: ignore
import dash  # type: ignore
import dash_bootstrap_components as dbc  # type: ignore
from dash import dcc, html, Input, Output, State, callback  # type: ignore
import plotly.graph_objects as go  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Colour palette — dark financial terminal
# ------------------------------------------------------------------
COLORS = {
    "bg": "#0D1117",
    "card_bg": "#161B22",
    "border": "#30363D",
    "green": "#00FF9C",
    "red": "#FF4B4B",
    "yellow": "#FFD700",
    "text": "#E6EDF3",
    "muted": "#8B949E",
    "blue": "#58A6FF",
}

API_BASE = "http://localhost:8000"

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

TICKER_OPTIONS = [
    {"label": "AAPL — Apple", "value": "AAPL"},
    {"label": "TSLA — Tesla", "value": "TSLA"},
    {"label": "MSFT — Microsoft", "value": "MSFT"},
    {"label": "AMZN — Amazon", "value": "AMZN"},
    {"label": "GOOGL — Alphabet", "value": "GOOGL"},
    {"label": "NVDA — Nvidia", "value": "NVDA"},
    {"label": "META — Meta", "value": "META"},
    {"label": "NFLX — Netflix", "value": "NFLX"},
    {"label": "BABA — Alibaba", "value": "BABA"},
    {"label": "UBER — Uber", "value": "UBER"},
]


def _fetch(endpoint: str) -> dict | list | None:
    """GET from the FastAPI backend; return None on any failure."""
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("API fetch failed for %s: %s", endpoint, exc)
        return None


def _signal_color(signal: str) -> str:
    """Map signal string to terminal colour."""
    mapping = {"Bullish": COLORS["green"], "Bearish": COLORS["red"]}
    return mapping.get(signal, COLORS["yellow"])


def _sentiment_badge(sentiment: str) -> html.Span:
    """Return a coloured badge element for a sentiment label."""
    color_map = {
        "positive": COLORS["green"],
        "negative": COLORS["red"],
        "neutral": COLORS["muted"],
    }
    color = color_map.get(sentiment.lower(), COLORS["muted"])
    label = sentiment.capitalize()
    return html.Span(
        label,
        style={
            "backgroundColor": color,
            "color": COLORS["bg"],
            "padding": "2px 8px",
            "borderRadius": "4px",
            "fontSize": "11px",
            "fontWeight": "700",
            "letterSpacing": "0.5px",
        },
    )


# ------------------------------------------------------------------
# App layout helpers
# ------------------------------------------------------------------

def _card(children, style: dict | None = None) -> html.Div:
    """Reusable card component in the terminal card style."""
    base = {
        "backgroundColor": COLORS["card_bg"],
        "border": f"1px solid {COLORS['border']}",
        "borderRadius": "8px",
        "padding": "16px",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)


def _metric_card(label: str, value_id: str, default: str = "—") -> html.Div:
    """Single metric display card."""
    return _card(
        [
            html.P(
                label,
                style={
                    "color": COLORS["muted"],
                    "fontSize": "11px",
                    "letterSpacing": "1px",
                    "textTransform": "uppercase",
                    "margin": "0 0 8px 0",
                },
            ),
            html.H3(
                default,
                id=value_id,
                style={
                    "color": COLORS["text"],
                    "fontSize": "26px",
                    "fontWeight": "700",
                    "margin": "0",
                },
            ),
        ],
        style={"flex": "1", "minWidth": "140px"},
    )


# ------------------------------------------------------------------
# App initialisation
# ------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="SentimentLens",
    suppress_callback_exceptions=True,
)
server = app.server  # expose for deployment

# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------

app.layout = html.Div(
    style={
        "backgroundColor": COLORS["bg"],
        "minHeight": "100vh",
        "fontFamily": "'Courier New', 'Consolas', monospace",
        "color": COLORS["text"],
        "padding": "0",
    },
    children=[
        # ── 1. Header ────────────────────────────────────────────────
        html.Div(
            style={
                "backgroundColor": COLORS["card_bg"],
                "borderBottom": f"1px solid {COLORS['border']}",
                "padding": "18px 32px",
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
            },
            children=[
                html.Div(
                    [
                        html.H1(
                            "SentimentLens",
                            style={
                                "color": COLORS["green"],
                                "fontSize": "28px",
                                "fontWeight": "700",
                                "margin": "0",
                                "letterSpacing": "2px",
                            },
                        ),
                    ]
                ),
                html.Div(
                    "Real-time AI sentiment for financial markets",
                    style={
                        "color": COLORS["muted"],
                        "fontSize": "13px",
                        "letterSpacing": "1px",
                    },
                ),
            ],
        ),

        # ── Main content wrapper ─────────────────────────────────────
        html.Div(
            style={"padding": "24px 32px"},
            children=[
                # ── 2. Ticker selector row ───────────────────────────
                html.Div(
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "16px",
                        "marginBottom": "24px",
                        "flexWrap": "wrap",
                    },
                    children=[
                        dcc.Dropdown(
                            id="ticker-dropdown",
                            options=TICKER_OPTIONS,
                            value="AAPL",
                            clearable=False,
                            style={
                                "width": "280px",
                                "backgroundColor": COLORS["card_bg"],
                                "color": COLORS["text"],
                                "border": f"1px solid {COLORS['border']}",
                                "borderRadius": "6px",
                            },
                        ),
                        html.Button(
                            "▶  Analyse",
                            id="analyse-btn",
                            n_clicks=0,
                            style={
                                "backgroundColor": COLORS["green"],
                                "color": COLORS["bg"],
                                "border": "none",
                                "borderRadius": "6px",
                                "padding": "8px 20px",
                                "fontFamily": "inherit",
                                "fontSize": "13px",
                                "fontWeight": "700",
                                "cursor": "pointer",
                                "letterSpacing": "1px",
                            },
                        ),
                        html.Span(
                            id="last-updated",
                            style={"color": COLORS["muted"], "fontSize": "12px"},
                        ),
                    ],
                ),

                # Loading wrapper for all dashboard content
                dcc.Loading(
                    id="loading-main",
                    type="dot",
                    color=COLORS["green"],
                    children=[
                        # ── 3. Metric cards ──────────────────────────
                        html.Div(
                            style={
                                "display": "flex",
                                "gap": "16px",
                                "marginBottom": "24px",
                                "flexWrap": "wrap",
                            },
                            children=[
                                _metric_card("Overall Signal", "metric-signal"),
                                _metric_card("Confidence", "metric-confidence"),
                                _metric_card("Articles Analysed", "metric-articles"),
                                _metric_card("Price Change (7d)", "metric-price-change"),
                            ],
                        ),

                        # ── 4. Charts row ────────────────────────────
                        html.Div(
                            style={
                                "display": "flex",
                                "gap": "16px",
                                "marginBottom": "24px",
                                "flexWrap": "wrap",
                            },
                            children=[
                                _card(
                                    [
                                        html.P(
                                            "Sentiment Trend (7 days)",
                                            style={
                                                "color": COLORS["muted"],
                                                "fontSize": "11px",
                                                "letterSpacing": "1px",
                                                "textTransform": "uppercase",
                                                "margin": "0 0 12px 0",
                                            },
                                        ),
                                        dcc.Loading(
                                            dcc.Graph(
                                                id="sentiment-chart",
                                                config={"displayModeBar": False},
                                                style={"height": "260px"},
                                            ),
                                            color=COLORS["green"],
                                        ),
                                    ],
                                    style={"flex": "1", "minWidth": "320px"},
                                ),
                                _card(
                                    [
                                        html.P(
                                            "Stock Price (7 days)",
                                            style={
                                                "color": COLORS["muted"],
                                                "fontSize": "11px",
                                                "letterSpacing": "1px",
                                                "textTransform": "uppercase",
                                                "margin": "0 0 12px 0",
                                            },
                                        ),
                                        dcc.Loading(
                                            dcc.Graph(
                                                id="price-chart",
                                                config={"displayModeBar": False},
                                                style={"height": "260px"},
                                            ),
                                            color=COLORS["green"],
                                        ),
                                    ],
                                    style={"flex": "1", "minWidth": "320px"},
                                ),
                            ],
                        ),

                        # ── 5. News feed ─────────────────────────────
                        _card(
                            [
                                html.P(
                                    "Latest News & Sentiment",
                                    style={
                                        "color": COLORS["text"],
                                        "fontSize": "14px",
                                        "fontWeight": "700",
                                        "letterSpacing": "1px",
                                        "margin": "0 0 16px 0",
                                    },
                                ),
                                html.Div(id="news-table"),
                            ],
                            style={"marginBottom": "24px"},
                        ),
                    ],
                ),
            ],
        ),

        # ── 6. Footer ────────────────────────────────────────────────
        html.Div(
            "Built by Omana Prabhakar · AI Product Builder · London · "
            "Extended from MSc thesis research at Heriot-Watt University",
            style={
                "textAlign": "center",
                "color": COLORS["muted"],
                "fontSize": "11px",
                "padding": "20px 32px",
                "borderTop": f"1px solid {COLORS['border']}",
                "letterSpacing": "0.5px",
            },
        ),

        # Auto-refresh every 5 minutes (300 000 ms)
        dcc.Interval(id="auto-refresh", interval=300_000, n_intervals=0),

        # Hidden store to hold the full analysis payload
        dcc.Store(id="analysis-store"),
        dcc.Store(id="price-store"),
    ],
)


# ------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------


@app.callback(
    Output("analysis-store", "data"),
    Output("price-store", "data"),
    Output("last-updated", "children"),
    Input("analyse-btn", "n_clicks"),
    Input("auto-refresh", "n_intervals"),
    State("ticker-dropdown", "value"),
    prevent_initial_call=False,
)
def fetch_data(n_clicks: int, n_intervals: int, ticker: str):
    """Fetch analysis and price data from the FastAPI backend."""
    analysis = _fetch(f"/analyse/{ticker}")
    price = _fetch(f"/price/{ticker}")
    ts = datetime.now(tz=timezone.utc).strftime("Last updated: %H:%M UTC")
    return analysis, price, ts


@app.callback(
    Output("metric-signal", "children"),
    Output("metric-signal", "style"),
    Output("metric-confidence", "children"),
    Output("metric-articles", "children"),
    Output("metric-price-change", "children"),
    Output("metric-price-change", "style"),
    Input("analysis-store", "data"),
    Input("price-store", "data"),
)
def update_metrics(analysis: dict | None, price: dict | None):
    """Populate the four metric cards."""
    base_style = {
        "color": COLORS["text"],
        "fontSize": "26px",
        "fontWeight": "700",
        "margin": "0",
    }

    signal = "—"
    signal_style = {**base_style, "color": COLORS["muted"]}
    confidence = "—"
    articles = "—"
    price_change = "—"
    price_style = {**base_style, "color": COLORS["muted"]}

    if analysis:
        agg = analysis.get("aggregate") or {}
        sig = agg.get("signal", "Neutral")
        signal = sig
        signal_style = {**base_style, "color": _signal_color(sig)}

        conf = agg.get("confidence", 0)
        confidence = f"{conf * 100:.1f}%"

        count = agg.get("article_count", 0)
        articles = str(count)

    if price:
        chg = price.get("change_pct", 0)
        price_change = f"{'+' if chg >= 0 else ''}{chg:.2f}%"
        price_color = COLORS["green"] if chg >= 0 else COLORS["red"]
        price_style = {**base_style, "color": price_color}

    return signal, signal_style, confidence, articles, price_change, price_style


@app.callback(
    Output("sentiment-chart", "figure"),
    Input("analysis-store", "data"),
)
def update_sentiment_chart(analysis: dict | None) -> go.Figure:
    """Render the 7-day sentiment trend line chart."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        font={"color": COLORS["text"], "family": "Courier New"},
        margin={"l": 40, "r": 20, "t": 10, "b": 40},
        xaxis={
            "gridcolor": COLORS["border"],
            "color": COLORS["muted"],
            "showline": False,
        },
        yaxis={
            "gridcolor": COLORS["border"],
            "color": COLORS["muted"],
            "range": [-1.1, 1.1],
            "zeroline": True,
            "zerolinecolor": COLORS["border"],
            "title": "Sentiment Score",
        },
        showlegend=False,
    )

    if not analysis:
        return fig

    trend = analysis.get("daily_trend") or []
    if not trend:
        return fig

    dates = [t["date"] for t in trend]
    scores = [t["avg_sentiment_score"] for t in trend]

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=scores,
            mode="lines+markers",
            line={"color": COLORS["green"], "width": 2},
            marker={"color": COLORS["green"], "size": 7},
            fill="tozeroy",
            fillcolor="rgba(0, 255, 156, 0.08)",
        )
    )

    return fig


@app.callback(
    Output("price-chart", "figure"),
    Input("price-store", "data"),
    State("ticker-dropdown", "value"),
)
def update_price_chart(price: dict | None, ticker: str) -> go.Figure:
    """Render the 7-day stock price line chart."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        font={"color": COLORS["text"], "family": "Courier New"},
        margin={"l": 40, "r": 20, "t": 10, "b": 40},
        xaxis={"gridcolor": COLORS["border"], "color": COLORS["muted"], "showline": False},
        yaxis={"gridcolor": COLORS["border"], "color": COLORS["muted"], "title": "Close Price (USD)"},
        showlegend=False,
    )

    if not price or not price.get("prices"):
        return fig

    dates = price.get("dates", [])
    prices = price.get("prices", [])
    chg = price.get("change_pct", 0)
    line_color = COLORS["green"] if chg >= 0 else COLORS["red"]

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=prices,
            mode="lines+markers",
            line={"color": line_color, "width": 2},
            marker={"color": line_color, "size": 6},
            name=ticker,
        )
    )

    return fig


@app.callback(
    Output("news-table", "children"),
    Input("analysis-store", "data"),
)
def update_news_table(analysis: dict | None) -> html.Div:
    """Render the news feed table."""
    if not analysis:
        return html.P(
            "Select a ticker and click Analyse.",
            style={"color": COLORS["muted"], "fontSize": "13px"},
        )

    articles: list[dict] = analysis.get("articles") or []
    if not articles:
        return html.P(
            "No articles found.",
            style={"color": COLORS["muted"], "fontSize": "13px"},
        )

    # Table header
    header_style = {
        "color": COLORS["muted"],
        "fontSize": "11px",
        "letterSpacing": "1px",
        "textTransform": "uppercase",
        "padding": "8px 12px",
        "borderBottom": f"1px solid {COLORS['border']}",
        "whiteSpace": "nowrap",
    }
    cell_style = {
        "padding": "10px 12px",
        "fontSize": "12px",
        "borderBottom": f"1px solid {COLORS['border']}",
        "verticalAlign": "middle",
    }

    header = html.Tr(
        [
            html.Th("Headline", style=header_style),
            html.Th("Source", style={**header_style, "width": "120px"}),
            html.Th("Date", style={**header_style, "width": "100px"}),
            html.Th("Sentiment", style={**header_style, "width": "90px"}),
            html.Th("Confidence", style={**header_style, "width": "90px"}),
        ]
    )

    rows = []
    for art in articles[:20]:  # Show up to 20 articles
        title = art.get("title") or ""
        truncated = title[:80] + ("…" if len(title) > 80 else "")
        pub = art.get("published_at") or ""
        date_str = pub[:10] if len(pub) >= 10 else "—"
        sentiment = art.get("sentiment") or "neutral"
        confidence = art.get("confidence") or 0.0
        source = art.get("source") or "—"

        rows.append(
            html.Tr(
                [
                    html.Td(
                        html.A(
                            truncated,
                            href=art.get("url") or "#",
                            target="_blank",
                            style={"color": COLORS["blue"], "textDecoration": "none"},
                        ),
                        style=cell_style,
                    ),
                    html.Td(source, style={**cell_style, "color": COLORS["muted"]}),
                    html.Td(date_str, style={**cell_style, "color": COLORS["muted"]}),
                    html.Td(_sentiment_badge(sentiment), style=cell_style),
                    html.Td(
                        f"{confidence * 100:.1f}%",
                        style={**cell_style, "color": COLORS["muted"]},
                    ),
                ],
                style={"transition": "background 0.15s"},
            )
        )

    return html.Table(
        [header] + rows,
        style={
            "width": "100%",
            "borderCollapse": "collapse",
            "fontSize": "12px",
        },
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
