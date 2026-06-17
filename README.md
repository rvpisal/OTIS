# OTIS — Options Trading Intelligence System

A Streamlit-based options screener that scans a configurable watchlist of liquid US equities, identifies technical setups, and surfaces actionable options strategies — all using free data from yfinance.

## Features

- **17 options strategies** — credit spreads, iron condors, iron butterflies, calendars, diagonals, debit spreads, covered calls, cash-secured puts, straddles, strangles, long calls/puts
- **Two-phase pipeline** — Phase 1 runs technical analysis on all tickers; Phase 2 fetches options chains only for tickers that triggered a strategy signal (keeps it fast)
- **Confidence scoring** — each setup is rated 0–100 based on trend alignment, RSI, IV rank, and Bollinger Band conditions
- **Legs column** — order-ticket ready strike/expiry layout per strategy (no ambiguous flat strike columns)
- **Single ticker lookup** — analyse any symbol on demand, bypassing the price gate
- **Configurable watchlist** — edit the 78-ticker default list directly in the sidebar
- **News & Events** — upcoming macro calendar (FOMC, CPI, NFP, PCE, GDP), live RSS headlines, per-ticker news, and earnings date warnings on every result row
- **Signal Backtester** — runs the exact same trigger logic over years of price history and shows win rates, average moves, and worst-case outcomes per strategy; includes a plain-English current-conditions summary

## Strategies covered

| Type | Strategies |
|---|---|
| Credit | Call Credit Spread, Put Credit Spread, Iron Condor, Iron Butterfly |
| Income | Cash-Secured Put, Covered Call, Short Strangle |
| Calendar / Diagonal | Call Calendar, Put Calendar, Call Diagonal, Put Diagonal |
| Debit (directional) | Bull Call Debit Spread, Bear Put Debit Spread |
| Long options | Long Call, Long Put, Long Straddle, Long Strangle |

## Setup

```bash
git clone https://github.com/rvpisal/OTIS.git
cd OTIS
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.11+. No API keys needed — all data is sourced from yfinance (free).

## Project structure

```
app.py              # Streamlit UI — pipeline orchestration, rendering, filters
data_fetcher.py     # yfinance wrappers — price history, options chains (Phase 1 & 2)
indicators.py       # Technical indicators + STRATEGY_TRIGGERS registry
strategy_matrix.py  # Options setup evaluation — strikes, legs, P&L metrics
backtester.py       # Historical signal backtest — forward-return stats per trigger
news_events.py      # Macro calendar, RSS headlines, ticker news, earnings dates
requirements.txt
```

## Design notes

- `STRATEGY_TRIGGERS` in `indicators.py` is the single source of truth for all entry conditions — used by both the live screener and the backtester to prevent drift.
- `STRATEGY_META` in `strategy_matrix.py` is the central registry for strategy metadata (name, emoji, color, bias, IV preference) used by the UI, confidence scorer, and backtester.
- Executability gates keep strikes realistic: `MAX_STRIKE_DIST = 15%` from spot + `MIN_LEG_BID = $0.05` prefilter before delta selection.
- Price gate: `$30` floor only — no upper cap, so mega-caps like META and LLY are included.
- Backtest is a **price-signal backtest** — free data has no historical options chains, so it validates entry signals against forward price returns, not actual spread P&L.

## Disclaimer

OTIS is a personal/educational research tool. Nothing here is financial advice. Options trading involves significant risk of loss.
