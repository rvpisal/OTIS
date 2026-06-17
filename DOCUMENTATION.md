# OTIS — Options Trading Intelligence System
## Full Technical Documentation

---

## Table of Contents

1. [What the App Does](#1-what-the-app-does)
2. [Tech Stack](#2-tech-stack)
3. [Architecture Overview](#3-architecture-overview)
4. [Data Flow](#4-data-flow)
5. [Module Reference](#5-module-reference)
   - [data_fetcher.py](#51-data_fetcherpy)
   - [indicators.py](#52-indicatorspy)
   - [strategy_matrix.py](#53-strategy_matrixpy)
   - [app.py](#54-apppy)
6. [Strategy Reference](#6-strategy-reference)
7. [Liquidity Gates](#7-liquidity-gates)
8. [Glossary](#8-glossary)

---

## 1. What the App Does

OTIS is an **end-of-day options trade screener** built entirely on free market data. It scans a curated watchlist of ~75 high-liquidity US stocks and ETFs, applies technical analysis to find setups worth investigating, and then screens live options chains to find specific spread trades that meet strict liquidity and risk/reward criteria.

**The core problem it solves:** Options traders need to manually scan dozens of tickers, check their technicals, pull options chains, calculate deltas, verify liquidity, and check spread economics — a process that takes hours. OTIS automates the entire workflow into a single button click.

**What it outputs:** Specific, actionable options spread setups with exact strikes, expiration dates, premium collected or paid, max profit, max loss, and liquidity quality indicators — for six different strategy types.

**What it does NOT do:** Execute trades, provide real-time data, or guarantee any financial outcome. It is an educational screening tool only.

---

## 2. Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **UI Framework** | [Streamlit](https://streamlit.io) ≥ 1.35 | Web dashboard, state management, interactive widgets |
| **Data Source** | [yfinance](https://github.com/ranaroussi/yfinance) ≥ 0.2.40 | Free EOD price history + live options chains from Yahoo Finance |
| **Data Processing** | [pandas](https://pandas.pydata.org) ≥ 2.1 | DataFrame manipulation for options chains and results |
| **Numerical Computing** | [numpy](https://numpy.org) ≥ 1.26 | Log return calculations, volatility math |
| **Technical Indicators** | [ta](https://github.com/bukosabino/ta) ≥ 0.11 | EMA, RSI, Bollinger Bands — pure Python, no C compilation |
| **Options Pricing (primary)** | [mibian](https://github.com/yassinemaaroufi/MibianLib) ≥ 0.1.3 | Black-Scholes delta via `mibian.BS()` |
| **Options Pricing (fallback)** | [py_vollib](https://github.com/vollib/py_vollib) ≥ 1.0.3 | Black-Scholes delta fallback if mibian fails |
| **Scientific Computing** | [scipy](https://scipy.org) ≥ 1.13 | Required dependency of mibian |
| **Runtime** | Python 3.12 (`.venv`) | Isolated virtual environment separate from system Python |
| **Persistence** | Python `pickle` (stdlib) | Disk-based results cache (`.otis_results.pkl`) |

**Why these choices:**
- `ta` over `TA-Lib`: No C compilation required — installs cleanly on any machine
- `mibian` + `py_vollib` with moneyness fallback: Three-tier resilience if IV data is missing
- `yfinance` only: Zero cost, no API keys needed
- `streamlit` session_state + pickle: Results survive browser tab close and back-navigation

---

## 3. Architecture Overview

The system is split into four modules with strict left-to-right data flow:

```
data_fetcher.py  →  indicators.py  →  strategy_matrix.py  →  app.py
    (Fetch)           (Analyze)           (Select)            (Display)
```

### Two-Phase Execution Design

The pipeline runs in two phases to minimize unnecessary API calls:

```
Phase 1 (all ~75 tickers):
  fetch_price_history() → analyze_ticker() → check_phase1_trigger()
                                                      │
                                           Only triggered tickers
                                                      ↓
Phase 2 (triggered tickers only):
  fetch_options_chain() → StrategySelector.run() → qualifying setups
```

**Why two phases?** Fetching an options chain takes ~2-3 seconds per ticker and is subject to rate limiting. In a trending market, 60-70% of tickers may not trigger any strategy. Skipping their options chains saves several minutes and reduces the chance of hitting Yahoo Finance rate limits.

---

## 4. Data Flow

```
TICKER_UNIVERSE (~75 tickers)
         │
         ▼
[Phase 1] DataFetcher.fetch_all_phase1()
  • Fetches 2 years of OHLCV price history per ticker (yfinance)
  • Applies price gate: $30–$500 (ETFs exempt from upper cap)
  • Returns: {ticker: {history: DataFrame, current_price: float}}
         │
         ▼
TechnicalAnalyzer.run_all()  [per ticker]
  • EMA-50, EMA-200 → macro_trend (BULLISH/BEARISH/NEUTRAL)
  • RSI(9), RSI(14)
  • Bollinger Bands (20-day, 2σ) → upper/lower touch flags
  • IV Rank (using historical volatility as proxy at this stage)
  • Returns: signals dict
         │
         ▼
check_phase1_trigger(signals)
  • Evaluates all 6 strategy trigger conditions
  • If ANY fire → ticker added to triggered_tickers list
         │
         ▼
[Phase 2] DataFetcher.fetch_options_for_triggered()
  • Fetches live options chains for triggered tickers only
  • Filters to standard monthly expirations (3rd Friday) only
  • DTE window: 14–75 days (covers all strategy needs)
  • Adds: expiration, dte, iv_pct columns
  • Returns: {ticker: {calls: DataFrame, puts: DataFrame}}
         │
         ▼
screen_triggered_tickers()
  • For each triggered ticker:
    - Extracts ATM IV from live options chain
    - Re-runs TechnicalAnalyzer with refined ATM IV → more accurate IV Rank
    - Runs StrategySelector.run() → evaluates all 6 strategies
  • Collects all qualifying setups into a single DataFrame
         │
         ▼
app.py (Streamlit)
  • Stores results in session_state + pickles to disk
  • Applies filter flags live on every slider change (no re-run needed)
  • Renders metrics, tabbed results table, signal detail expander
```

---

## 5. Module Reference

---

### 5.1 `data_fetcher.py`

Responsible for all data ingestion. Nothing else in the system calls yfinance directly.

---

#### Constants

```python
TICKER_UNIVERSE  # list[str] — ~75 pre-screened high-liquidity tickers
INDEX_ETFS       # set[str] — ETFs exempt from the $500 price cap
PRICE_MIN = 30.0
PRICE_MAX = 500.0
DTE_MIN = 14     # wide window covers all strategy DTE needs
DTE_MAX = 75
```

**TICKER_UNIVERSE** is a curated "Goldilocks" list of stocks and ETFs pre-validated to meet: market cap > $15B, avg daily share volume > 3M, avg daily option volume > 10,000 contracts. Divided into: Broad ETFs, Sector ETFs, Mega-cap Tech, Financials, Healthcare, Consumer/Retail, Energy, Industrials, Telecom, and other high-liquidity names.

---

#### `is_standard_monthly(date_str: str) -> bool`

Determines if a given expiration date string (`"YYYY-MM-DD"`) is a **standard monthly expiration**.

Handles two cases:
1. **Normal:** 3rd Friday of the month — `weekday() == 4` and `15 <= day <= 21`
2. **Holiday-adjusted:** Thursday before a 3rd Friday, when that Friday is a US market holiday (e.g., Juneteenth June 19 → expiry falls on Thursday June 18)

This filter ensures the screener only looks at standard monthly options with the deepest liquidity, not weeklies.

```python
is_third_friday = is_standard_monthly  # backwards-compatible alias
```

---

#### `_sleep() -> None`

Applies a random delay of 1.0–3.0 seconds before each yfinance API call. Prevents rate-limiting by Yahoo Finance when iterating many tickers sequentially.

---

#### `class DataFetcher`

Main ingestion class. Instantiated with a ticker list and history period.

```python
DataFetcher(tickers=TICKER_UNIVERSE, history_period="2y")
```

`history_period="2y"` ensures at least 200 trading days of data even on thin history years, satisfying the `MIN_HISTORY_ROWS=200` requirement in `indicators.py`.

---

##### `fetch_price_history(ticker: str) -> pd.DataFrame | None`

Fetches OHLCV (Open, High, Low, Close, Volume) daily price history for one ticker.

**Steps:**
1. Calls `_sleep()` to rate-limit
2. Calls `yf.Ticker(ticker).history(period="2y")`
3. Reads `df["Close"].iloc[-1]` as `last_close`
4. Applies the **price gate**: rejects if `last_close < $30` or (for non-ETFs) `last_close > $500`
5. Returns the DataFrame on success, `None` on any failure

**Why the ETF exemption?** SPY, QQQ, and similar ETFs legitimately trade above $500 (SPY > $700 in 2026) and have enormous option liquidity. Excluding them would be wrong.

---

##### `fetch_options_chain(ticker: str) -> dict | None`

Fetches live options chain data for one ticker, pre-filtered to qualifying expirations.

**Steps:**
1. Calls `_sleep()`
2. Gets all available expirations via `yf.Ticker(ticker).options`
3. For each expiration:
   - Skips if `is_standard_monthly()` returns False (filters out weeklies)
   - Skips if DTE is outside `[DTE_MIN=14, DTE_MAX=75]`
   - Fetches call and put chains via `t.option_chain(exp_str)`
   - Adds computed columns: `expiration` (str), `dte` (int), `iv_pct` (impliedVolatility × 100)
4. Concatenates all qualifying calls and puts
5. Returns `{"calls": DataFrame, "puts": DataFrame}` or `None`

The wide DTE window (14–75) intentionally covers all strategies:
- Credit spreads target 30–45 DTE (filtered later in `_select_expiration_window`)
- Calendar spreads need 15–30 DTE (front) and 45–65 DTE (back)

---

##### `fetch_all_phase1(progress_callback=None) -> dict`

Phase 1 driver. Iterates all tickers and calls `fetch_price_history()` for each.

**Returns:** `{ticker: {"history": DataFrame, "current_price": float}}` for tickers that passed all gates.

Accepts an optional `progress_callback(i, total, ticker)` used by `app.py` to update the progress bar.

---

##### `fetch_options_for_triggered(triggered_tickers: list, progress_callback=None) -> dict`

Phase 2 driver. Iterates only the triggered tickers and calls `fetch_options_chain()` for each.

**Returns:** `{ticker: options_data_or_None}`

---

### 5.2 `indicators.py`

Computes all technical indicators from price history and evaluates Phase 1 trigger conditions.

---

#### Constants

```python
MIN_HISTORY_ROWS = 200  # minimum rows needed to compute EMA-200 reliably
```

---

#### `class TechnicalAnalyzer`

Wraps a price history DataFrame and exposes indicator computation methods.

```python
TechnicalAnalyzer(df: pd.DataFrame)
# Raises ValueError if len(df) < MIN_HISTORY_ROWS
```

---

##### `compute_ema(window: int) -> pd.Series`

Computes Exponential Moving Average using the `ta` library.

```python
ta.trend.EMAIndicator(close=self.close, window=window).ema_indicator()
```

Used with windows 50 and 200 to determine macro trend.

---

##### `compute_rsi(window: int) -> float`

Computes Relative Strength Index for the given window, returning only the most recent value.

```python
ta.momentum.RSIIndicator(close=self.close, window=window).rsi().iloc[-1]
```

Called with windows 9 (for trigger signals) and 14 (for confirmation display).

---

##### `compute_bollinger_bands(window=20, window_dev=2) -> dict`

Computes 20-day Bollinger Bands (±2 standard deviations around the 20-day SMA).

**Returns:**
```python
{
    "upper": float,        # upper band value
    "lower": float,        # lower band value
    "mid": float,          # 20-day SMA (middle band)
    "upper_touch": bool,   # True if last close >= upper band
    "lower_touch": bool,   # True if last close <= lower band
}
```

A "touch" signals price is statistically over-extended and due for mean reversion.

---

##### `compute_hv_and_iv_rank(current_atm_iv: float | None = None) -> dict`

Calculates Historical Volatility and IV Rank.

**HV calculation:**
```
log_returns = ln(Close[t] / Close[t-1])
hv_21_annualized = log_returns.rolling(21).std() × √252
```

**IV Rank formula:**
```
IV Rank = (Current IV − 52-wk Min HV) / (52-wk Max HV − 52-wk Min HV) × 100
```

- `current_atm_iv`: ATM implied vol from the live options chain (decimal, e.g. 0.35). If `None`, uses the latest 21-day HV as a proxy.
- Guard: if max == min (zero-vol asset), returns `iv_rank = 50.0`

**Returns:**
```python
{
    "hv_current": float,    # today's 21-day annualized HV
    "hv_252_min": float,    # rolling minimum over 252 days
    "hv_252_max": float,    # rolling maximum over 252 days
    "iv_rank": float,       # 0–100
    "high_vol_env": bool,   # True if iv_rank > 45
}
```

---

##### `compute_macro_trend(ema50: float, ema200: float) -> str`

Determines the macro directional trend by comparing EMA-50 to EMA-200.

```
threshold = ema200 × 0.005  (0.5% buffer to avoid noise)

EMA50 > EMA200 + threshold  → "BULLISH"
EMA50 < EMA200 - threshold  → "BEARISH"
otherwise                   → "NEUTRAL"
```

The 0.5% buffer prevents frequent BULLISH/BEARISH flipping when EMAs are nearly equal.

---

##### `run_all(current_atm_iv: float | None = None) -> dict | None`

Orchestrates all indicator computations and returns a flat **signal bundle** dict.

**Returns `None`** on any exception (defensive — ensures one bad ticker doesn't crash the entire pipeline).

**Signal bundle keys:**
```python
{
    "macro_trend": str,       # "BULLISH" | "BEARISH" | "NEUTRAL"
    "ema50": float,
    "ema200": float,
    "rsi9": float,
    "rsi14": float,
    "rsi9_overbought": bool,  # rsi9 > 70
    "rsi9_oversold": bool,    # rsi9 < 30
    "bb_upper": float,
    "bb_lower": float,
    "bb_mid": float,
    "bb_upper_touch": bool,
    "bb_lower_touch": bool,
    "hv_current": float,
    "iv_rank": float,
    "high_vol_env": bool,     # iv_rank > 45
    "current_price": float,
}
```

---

#### `has_any_trigger(signals: dict) -> bool`

The Phase 1 gate. Returns `True` if ANY strategy trigger condition is satisfied for this ticker.

**Trigger conditions evaluated (any one is sufficient):**

| Strategy | Condition |
|----------|-----------|
| Call Credit Spread (Group A) | `macro_trend in ("BEARISH", "NEUTRAL")` |
| Call Credit Spread (Group B) | `macro_trend == "BULLISH"` AND `rsi9 > 72` AND `bb_upper_touch` AND `high_vol_env` |
| Put Credit Spread | `macro_trend == "BULLISH"` AND `rsi9 < 28` AND `bb_lower_touch` AND `high_vol_env` |
| Call Calendar Spread | `macro_trend in ("BULLISH", "NEUTRAL")` AND `38 ≤ rsi9 ≤ 62` AND `15 ≤ iv_rank ≤ 40` AND no BB touch AND `|price - ema50| / ema50 ≤ 3%` |
| Bull Call Debit Spread | `macro_trend == "BULLISH"` AND `50 ≤ rsi9 ≤ 75` AND `iv_rank < 50` |
| Bear Put Debit Spread | `macro_trend in ("BEARISH", "NEUTRAL")` AND `25 ≤ rsi9 ≤ 50` AND `iv_rank < 50` |

If none fire → no options chain is fetched for this ticker (saves API calls and time).

---

#### `analyze_ticker(ticker, history_df, current_atm_iv=None) -> dict | None`

Convenience wrapper. Instantiates `TechnicalAnalyzer` and calls `run_all()`. Returns `None` if fewer than `MIN_HISTORY_ROWS` rows or any exception occurs.

---

#### `check_phase1_trigger(signals: dict | None) -> bool`

Convenience wrapper around `has_any_trigger()` that safely handles `None` signals (returns `False`).

---

### 5.3 `strategy_matrix.py`

Contains all four liquidity gate functions, the delta approximation engine, and the `StrategySelector` class that evaluates all six strategy types.

---

#### Constants

```python
DELTA_LOW = 0.15      # minimum acceptable delta for credit spread short legs
DELTA_HIGH = 0.20     # maximum acceptable delta for credit spread short legs
DELTA_TARGET = 0.175  # ideal delta (pick the candidate closest to this)
SPREAD_WIDTH_RANGE = range(1, 6)  # search widths $1 through $5
MIN_OI = 1_000        # minimum open interest for primary legs
MIN_CREDIT_PCT = 0.20 # minimum credit as fraction of spread width (20%)
```

---

#### `_moneyness_delta(flag: str, S: float, K: float) -> float`

Last-resort delta approximation using only the stock price (S) to strike (K) ratio. Used when no IV data is available and both mibian and py_vollib fail.

```
moneyness = S / K

Calls:
  m > 1.05  → 0.80  (deep ITM)
  m > 1.00  → 0.55  (slightly ITM)
  m > 0.97  → 0.40  (near ATM)
  m > 0.93  → 0.20  (OTM)
  else      → 0.10  (deep OTM)

Puts: negative mirror of call (approximate put-call parity)
```

---

#### `get_delta(flag, S, K, iv_pct, dte, r=0.05) -> float`

Black-Scholes delta with **three-tier fallback**:

1. **mibian** (`mibian.BS([S, K, r×100, dte], volatility=iv_pct)`) — most accurate
2. **py_vollib** (`bs_delta(flag, S, K, dte/365, r, iv_pct/100)`) — fallback if mibian fails
3. **Moneyness proxy** (`_moneyness_delta`) — last resort when IV is zero/missing

Always returns a float and never raises. `flag` is `"c"` for calls, `"p"` for puts.

---

#### `passes_bidask_gate(leg: pd.Series, ticker: str) -> bool`

**Gate 2:** Checks if a single option contract's bid/ask spread is within acceptable bounds.

```
mid = (bid + ask) / 2

For INDEX_ETFS:  (ask - bid) ≤ $0.05  (flat dollar cap)
For equities:    (ask - bid) / mid ≤ 10%
```

ETFs get a flat cap because their options are so liquid that any spread above $0.05 is already meaningful slippage.

---

#### `passes_oi_gate(leg: pd.Series, min_oi=1000) -> bool`

**Gate 3:** Checks if open interest meets the minimum threshold.

```
openInterest >= 1,000
```

Applied to the primary leg of every strategy. Ensures the position can be entered and exited without meaningful market impact.

---

#### `passes_yield_gate(credit, spread_width, min_pct=0.20) -> bool`

**Gate 4:** Checks if the net credit collected meets the minimum yield-to-risk ratio.

```
credit >= min_pct × spread_width
```

Default: credit must be ≥ 20% of the spread width. On a $5-wide spread, you must collect ≥ $1.00 per share ($100 per contract). This ensures the risk/reward is mathematically worthwhile relative to margin requirements.

---

#### `class StrategySelector`

The core evaluation engine. One instance per ticker per pipeline run.

```python
StrategySelector(
    ticker: str,
    signals: dict,          # signal bundle from TechnicalAnalyzer
    options_data: dict,     # {"calls": DataFrame, "puts": DataFrame}
    min_credit_pct=0.20,    # yield gate threshold
    r=0.05,                 # risk-free rate for Black-Scholes
)
```

`self.S = signals["current_price"]` — stored as the spot price reference for delta calculations.

---

##### `_select_expiration_window(chain_df) -> pd.DataFrame`

Filters the options chain to the target DTE window for credit spreads.

**Primary target:** 30–45 DTE (optimal theta/vega tradeoff).

**Fallback:** If no standard monthly expiration lands in [30, 45] (e.g., holiday-adjusted June expiry is 29 DTE), searches [21, 60] and picks the expiration **closest to 37 DTE** (the midpoint). This prevents silent failures when the calendar shifts expirations just outside the ideal window.

---

##### `_find_short_leg(chain_df, flag, delta_low, delta_high, delta_target, require_oi=True) -> pd.Series | None`

The primary strike-selection engine. Finds the best single option row that meets all three criteria: delta range, bid/ask gate, and (optionally) OI gate.

**Algorithm:**
1. Iterates every row in `chain_df`
2. Computes delta using `get_delta()` with the three-tier fallback
3. Rejects if `|delta|` is outside `[delta_low, delta_high]`
4. Rejects if `passes_bidask_gate()` fails
5. Rejects if `require_oi=True` and `passes_oi_gate()` fails
6. Stores passing candidates as `(distance_from_target, row)` tuples
7. Sorts by distance and returns the row closest to `delta_target`

Returns `None` if no candidates pass all gates.

**Used with different delta parameters for different roles:**
- Credit spread short leg: `[0.15, 0.20]` target 0.175 (low probability of loss)
- Debit spread long leg: `[0.35, 0.65]` target 0.50 (near ATM, high directional exposure)
- Debit spread short leg: `[0.15, 0.35]` target 0.25 (OTM, caps max profit but reduces cost)

---

##### `_find_long_leg(chain_df, short_strike, flag, short_bid) -> pd.Series | None`

Finds the hedge (long) leg of a **credit spread** by searching outward from the short strike in $1 increments up to $5.

**For each width from $1 to $5:**
1. Targets `short_strike + width` (calls) or `short_strike - width` (puts)
2. Finds the closest available strike
3. Applies `passes_bidask_gate()` on the long leg
4. Computes `credit = short_bid - long_ask`
5. Applies `passes_yield_gate(credit, actual_width, self.min_credit_pct)`
6. Returns the first width that passes all checks (narrowest valid spread)

---

##### `_compute_credit(short_leg, long_leg) -> float | None`

Computes the net credit for a credit spread using conservative pricing:

```
credit = short_leg.bid - long_leg.ask
```

Using `bid` for the short (what you sell at) and `ask` for the long (what you buy at) reflects realistic fill prices (worst-case scenario, not mid). Returns `None` if the result is zero or negative (inverted spread).

---

##### `_get_atm_iv() -> float | None`

Extracts the current ATM implied volatility from the live options chain.

**Algorithm:**
1. Looks at both calls and puts in the 30–45 DTE window
2. For each, finds the strike closest to `self.S` (current price)
3. Reads `iv_pct` (impliedVolatility × 100) at that strike
4. Returns the average of call and put ATM IV as a decimal (e.g., 0.35 for 35%)

This value feeds back into `indicators.py` to compute a more accurate IV Rank than the pure HV proxy used in Phase 1.

---

##### `_find_calendar_legs(calls_df) -> tuple[pd.Series | None, pd.Series | None]`

Finds both legs of a call calendar spread.

**Front month (short leg):**
- DTE in [15, 30]
- Strike closest to `self.S` (ATM)
- Must pass `passes_bidask_gate()` and `passes_oi_gate()`

**Back month (long leg):**
- DTE in [45, 65]
- Same strike as front month (or nearest available)
- Must pass `passes_bidask_gate()` (no OI gate required — you're buying this leg and can always exit via the bid)

Returns `(None, None)` if either leg fails.

---

##### `_build_result(strategy, short_leg, long_leg, credit, short_delta) -> dict`

Constructs the standardized result dict for credit spread strategies (CCS, PCS).

Computes:
- `spread_width = |long_strike - short_strike|`
- `credit_pct = credit / spread_width × 100`
- `max_profit = credit × 100` (per contract)
- `max_loss = (spread_width - credit) × 100` (per contract)
- `liquidity_warning = not passes_bidask_gate(short_leg)` (redundant safety check)

---

##### `evaluate_call_credit_spread() -> dict | None`

**Strategy:** Sell an OTM call, buy a higher-strike call as a hedge. Net credit collected.

**Profit condition:** Stock closes below short strike at expiration.

**Trigger conditions (Group A OR Group B):**
- **Group A:** `macro_trend in ("BEARISH", "NEUTRAL")` — any non-bullish stock qualifies
- **Group B:** `macro_trend == "BULLISH"` AND `rsi9 > 72` AND `bb_upper_touch` AND `high_vol_env` — overbought breakout in a bull trend

**Execution:**
1. Checks trigger
2. Calls `_select_expiration_window(calls)` → 30–45 DTE window
3. Calls `_find_short_leg(calls, "c")` → delta [0.15, 0.20] + Gate 2 + Gate 3
4. Calls `_find_long_leg(calls, short_strike, "c", short_bid)` → Gate 2 + Gate 4
5. Computes credit and validates with `passes_yield_gate()`
6. Calls `_build_result("CALL_CREDIT_SPREAD", ...)`

---

##### `evaluate_put_credit_spread() -> dict | None`

**Strategy:** Sell an OTM put, buy a lower-strike put as a hedge. Net credit collected.

**Profit condition:** Stock closes above short strike at expiration.

**Trigger (all conditions required):**
- `macro_trend == "BULLISH"` AND `rsi9 < 28` AND `bb_lower_touch` AND `high_vol_env`

Requires the stock to be in a bull trend but oversold — a mean-reversion bet that the pullback is temporary.

**Execution:** Same gate sequence as CCS but on the puts chain.

---

##### `evaluate_iron_condor(ccs_result, pcs_result) -> dict | None`

**Strategy:** Both a Call Credit Spread and a Put Credit Spread on the same ticker simultaneously, creating an upper and lower boundary. Net credit collected.

**Profit condition:** Stock stays between both short strikes at expiration.

**Pre-condition:** Both `ccs_result` and `pcs_result` must be non-None (both legs independently qualified).

**Safety guard:** `pcs_result["short_strike"] < ccs_result["short_strike"]` — put short must be strictly below call short, otherwise the condor is inverted (worthless).

**Combined metrics:**
```
combined_credit = ccs_credit + pcs_credit
wider_width = max(ccs_width, pcs_width)
max_loss = (wider_width - combined_credit) × 100
```

---

##### `evaluate_call_calendar_spread() -> dict | None`

**Strategy:** Sell a near-term ATM call (front month), buy the same strike in a later month (back month). Net debit paid.

**Profit condition:** Stock stays near the ATM strike at front-month expiry, allowing the front-month to decay faster than the back-month (theta differential profit).

**Trigger (all required):**
- `macro_trend in ("BULLISH", "NEUTRAL")`
- `38 ≤ rsi9 ≤ 62` — genuinely neutral momentum
- `15 ≤ iv_rank ≤ 40` — LOW IV is critical: calendars are long-vega, so entering in high IV means overpaying for the back month
- No Bollinger Band touch — price must not be trending strongly
- `|price - ema50| / ema50 ≤ 3%` — stock must be coiling near fair value

**Net debit gate:** `net_debit ≤ 0.66 × back_month_ask` (ensures you're not overpaying relative to the back month's value).

**Calendar-specific result fields:** `net_debit`, `front_expiration`, `front_dte`, `back_expiration`, `back_dte`

---

##### `evaluate_bull_call_debit_spread() -> dict | None`

**Strategy:** Buy a near-ATM call, sell a higher-strike OTM call at the same expiry. Net debit paid.

**Profit condition:** Stock rises above the long strike toward the short strike at expiry.

**Trigger (all required):**
- `macro_trend == "BULLISH"` — directional bet requires trend confirmation
- `50 ≤ rsi9 ≤ 75` — upward momentum present but not at overbought extreme
- `iv_rank < 50` — lower IV means cheaper options = smaller debit to pay

**Leg selection:**
- Long leg (buy): `_find_short_leg(calls, "c", delta_low=0.35, delta_high=0.65, delta_target=0.50, require_oi=True)` — near-ATM, OI gate required (need liquidity to exit)
- Short leg (sell): `_find_short_leg(otm_calls, "c", delta_low=0.15, delta_high=0.35, delta_target=0.25, require_oi=False)` — OTM, caps max profit

**Debit gate:** `net_debit < 0.75 × spread_width` — ensures at least 25% profit potential relative to cost.

---

##### `evaluate_bear_put_debit_spread() -> dict | None`

**Strategy:** Buy a near-ATM put, sell a lower-strike OTM put at the same expiry. Net debit paid.

**Profit condition:** Stock falls below the long strike toward the short strike at expiry.

**Trigger (all required):**
- `macro_trend in ("BEARISH", "NEUTRAL")` — bearish directional setup
- `25 ≤ rsi9 ≤ 50` — downward momentum present
- `iv_rank < 50` — cheaper premiums = smaller debit

**Leg selection:** Same as Bull Call Debit but on puts chain, with the short leg below (OTM put) the long leg.

---

##### `run() -> list[dict]`

Orchestrates all six strategy evaluations for one ticker. Returns a list of 0–6 result dicts (one per qualifying strategy found). Empty list if nothing qualifies.

```python
ccs = self.evaluate_call_credit_spread()
pcs = self.evaluate_put_credit_spread()
ic  = self.evaluate_iron_condor(ccs, pcs)
cal = self.evaluate_call_calendar_spread()
bcs = self.evaluate_bull_call_debit_spread()
bps = self.evaluate_bear_put_debit_spread()
return [r for r in (ccs, pcs, ic, cal, bcs, bps) if r is not None]
```

Note: Iron Condor is derived from CCS + PCS results — it requires no additional data fetch.

---

#### `screen_triggered_tickers(phase1_data, phase2_options, signals_data, r, min_credit_pct) -> pd.DataFrame`

Module-level entry point called by `app.py`.

**For each triggered ticker:**
1. Builds a temporary `StrategySelector` and calls `_get_atm_iv()` to extract ATM IV
2. Re-runs `analyze_ticker()` with the refined ATM IV for a more accurate IV Rank
3. Creates the final `StrategySelector` and calls `run()`
4. Collects all results

Returns a single concatenated `pd.DataFrame` of all qualifying setups across all tickers. Empty DataFrame if nothing qualifies.

**Important:** `min_credit_pct=0.0` is always passed from `app.py` — all setups with positive credit are stored, and the yield filter is applied live by `add_filter_flags()` in the UI without requiring a pipeline re-run.

---

### 5.4 `app.py`

The Streamlit dashboard. Entry point via `streamlit run app.py`.

---

#### Constants

```python
CACHE_FILE = Path(__file__).parent / ".otis_results.pkl"
```

Disk cache file stored in the project directory. Survives browser tab close and back-navigation.

---

#### `_H` dict

Centralized help text strings for all sidebar widgets, metrics, and signal detail panels. Kept in one place so updates to copy don't require hunting through UI code. Referenced as `help=_H["key"]` throughout.

---

#### `_DEBIT_STRATEGIES` frozenset

```python
frozenset({"CALL_CALENDAR_SPREAD", "BULL_CALL_DEBIT_SPREAD", "BEAR_PUT_DEBIT_SPREAD"})
```

Used throughout filter logic to identify strategies that pay a debit (no credit% to compare against) so the credit% filter is correctly skipped for them.

---

#### `_STRATEGY_INFO` dict

Per-strategy markdown description strings shown as info banners in each results tab. Contains plain-English explanation of what the strategy is, when it profits, and what technical conditions triggered it.

---

#### `save_cache(state: dict) -> None`

Saves the current pipeline results to disk as a pickle file.

**Persists:** `results_df`, `signals_data`, `triggered_tickers`, `total_tickers`, `last_run`

Called automatically after every successful pipeline run. Silently swallows exceptions (disk full, permissions) so a cache failure never crashes the app.

---

#### `load_cache() -> dict | None`

Loads the pickle cache from disk.

Returns the payload dict with `from_cache=True` added, or `None` if the file doesn't exist or is corrupted. Called once on app startup when `session_state` has no results yet.

---

#### `render_sidebar() -> dict`

Renders the entire left sidebar with all filter controls and the Quick Glossary expander.

**Widgets rendered:**
| Widget | Type | Default | Purpose |
|--------|------|---------|---------|
| Min IV Rank | Slider (0–100, step 5) | 45 | Minimum IV Rank for a setup to match filters |
| Min Credit % | Slider (10–50, step 5) | 20 | Minimum yield-to-risk ratio for credit spreads |
| DTE Min | Number input (1–60) | 30 | Minimum days to expiration |
| DTE Max | Number input (1–60) | 45 | Maximum days to expiration |
| Strategies | Multiselect | All 6 | Strategy types to show |
| Risk-Free Rate % | Slider (0–10, step 0.25) | 5.0 | For Black-Scholes delta calculation |
| 📖 Quick Glossary | Expander | collapsed | Defines key terms |
| 🗑️ Clear Saved Results | Button | shown if cache exists | Wipes disk cache and resets session |

**Returns:** A `filters` dict consumed by every rendering function that needs to apply or display filter criteria.

---

#### `run_full_pipeline(tickers, r, phase1_bar, phase2_bar, status_text) -> tuple`

Orchestrates the complete two-phase data pipeline with live progress reporting.

**Steps:**
1. Instantiates `DataFetcher`
2. Calls `fetch_all_phase1()` → updates `phase1_bar` via callback
3. Loops over tickers: `analyze_ticker()` → `check_phase1_trigger()` → builds `triggered_tickers`
4. Calls `fetch_options_for_triggered()` → updates `phase2_bar` via callback
5. Calls `screen_triggered_tickers()` with `min_credit_pct=0.0`
6. Updates `status_text` throughout

**Returns:** `(phase1_data, signals_data, triggered_tickers, results_df)`

---

#### `_row_passes(row, filters) -> bool`

Internal helper. Returns `True` if a single result row satisfies all current sidebar filters.

Checks: IV Rank ≥ min, credit% ≥ min (skipped for debit strategies), DTE in range, strategy type in selected list.

---

#### `add_filter_flags(df, filters) -> pd.DataFrame`

Adds a `filter_status` column to every row in the results DataFrame.

- **`"✅ Matches filters"`** — row passes all current sidebar criteria
- **`"⚠️ IV Rank 38 < 45 · Credit 14.2% < 20%"`** — fails one or more, with exact values shown

This is the key function enabling the "show everything, highlight what's different" UX. Every sidebar slider change re-triggers a Streamlit re-render which calls this function fresh — no pipeline re-run needed.

---

#### `count_matching(df, filters) -> int`

Returns the count of rows that pass all current sidebar filters. Used in the metrics row to show "Match Current Filters" alongside the total setups found.

---

#### `style_results_df(df) -> Styler`

Produces a fully styled pandas `Styler` object for `st.dataframe()`.

**Columns displayed (in order):**
`Filter Status | Ticker | Price | Trend | RSI(9) | IV Rank | Strategy | Strike (Long/Buy) | Strike (Short/Sell) | Expiry | DTE | Primary OI | Credit | Net Debit | Credit % | Debit % | Max Loss`

**Styling applied:**
- `row_bg()`: Row background by filter status and strategy type
  - ⚠️ rows → grey (`#ebebeb`)
  - ✅ CCS rows → red (`#f8d7da`)
  - ✅ PCS rows → green (`#d4edda`)
  - ✅ IC rows → yellow (`#fff3cd`)
  - ✅ Calendar rows → blue (`#dce8f8`)
  - ✅ Bull Call Debit rows → amber (`#fff0dc`)
  - ✅ Bear Put Debit rows → purple (`#f0e6ff`)
- `filter_status_cell()`: Green cell for ✅, amber cell for ⚠️
- `oi_cell()`: Orange cell for OI in 1,000–2,000 range (passed gate but low buffer)
- `credit`/`net_debit` columns mutually exclusive per row (shows `—` for the inapplicable one)
- `credit_pct`/`debit_pct` columns mutually exclusive per row

---

#### `render_metrics_row(results_df, total_tickers, triggered_count, filters)`

Renders the five summary metric cards at the top of the results section.

| Card | Value | Meaning |
|------|-------|---------|
| Tickers Screened | N | Passed Phase 1 price gate |
| Triggered | N | Had options chains fetched (Phase 2) |
| Setups Found | N | Total qualifying setups from pipeline |
| Match Current Filters | N | How many pass current sidebar sliders (live, no re-run) |
| Best Credit % | X.X% | Highest credit/width ratio among credit spread setups |

---

#### `render_column_guide()`

Renders a collapsible expander above the results table with a markdown table explaining every column's meaning. Includes row colour legend.

---

#### `render_signal_detail(ticker, signals_data)`

Renders the technical indicator breakdown for a selected ticker in a two-column layout.

**Left column — Trend & Momentum:**
- Macro Trend (with BULLISH/BEARISH/NEUTRAL label)
- EMA 50 and EMA 200 prices
- RSI(9) with OVERBOUGHT/OVERSOLD delta indicator
- RSI(14) for confirmation

**Right column — Volatility & Bands:**
- Current Price
- IV Rank with a `st.progress()` bar (0–100)
- High Vol Environment (YES/NO)
- Upper Bollinger Band (with TOUCHED indicator if active)
- Lower Bollinger Band (with TOUCHED indicator if active)

---

#### `render_results_table(results_df, filters, signals_data)`

Main results rendering function. Coordinates the full display.

**Steps:**
1. Calls `add_filter_flags()` to annotate every row
2. Shows a summary banner (e.g., "5 of 12 setups match your filters")
3. Calls `render_column_guide()`
4. Splits results into 7 strategy-specific DataFrames
5. Creates 7 tabs with labels showing `(matching/total)` count per strategy
6. For each tab: renders strategy description banner → `st.dataframe()` → selectbox for signal detail

---

#### `render_welcome()`

Shown only when no results exist yet (first visit or after clearing cache).

Displays:
- Instruction to click Run Screen
- "How It Works" — three cards explaining Phase 1, Phase 2, and strategies
- "The 4 Liquidity Gates" — four green success cards explaining each gate

---

#### `main()`

Streamlit entry point. Called on every page interaction.

**Execution sequence:**
1. Renders title and caption
2. Calls `render_sidebar()` → gets current `filters`
3. On first load: attempts `load_cache()` → populates `session_state`
4. Renders Run Screen button and cache notice
5. If button clicked: runs `run_full_pipeline()` → saves to `session_state` → calls `save_cache()`
6. If results exist: calls `render_metrics_row()` + `render_results_table()`
7. If no results: calls `render_welcome()`

---

## 6. Strategy Reference

| Strategy | Type | Trend | RSI(9) | IV Rank | Profit When | Max Profit | Max Loss |
|----------|------|-------|--------|---------|-------------|-----------|---------|
| 🔴 Call Credit Spread | Credit | Bearish/Neutral OR Overbought Bull | Any (Group A) or >72 (Group B) | Any (Group A) or >45 (Group B) | Stock stays below short strike | Credit × 100 | (Width − Credit) × 100 |
| 🟢 Put Credit Spread | Credit | Bullish | <28 | >45 | Stock stays above short strike | Credit × 100 | (Width − Credit) × 100 |
| 🟡 Iron Condor | Credit | Requires both CCS + PCS | Requires both | Requires both | Stock stays between both short strikes | Combined Credit × 100 | (Wider Width − Credit) × 100 |
| 🔵 Call Calendar | Debit | Bullish/Neutral | 38–62 | 15–40 | Stock stays near ATM strike at front expiry | Back month value − Net Debit | Net Debit × 100 |
| 🟠 Bull Call Debit | Debit | Bullish | 50–75 | <50 | Stock rises above long strike | (Width − Debit) × 100 | Net Debit × 100 |
| 🟣 Bear Put Debit | Debit | Bearish/Neutral | 25–50 | <50 | Stock falls below long strike | (Width − Debit) × 100 | Net Debit × 100 |

---

## 7. Liquidity Gates

Every setup must pass all applicable gates before appearing in results:

| Gate | What It Checks | Threshold | Why It Matters |
|------|---------------|-----------|----------------|
| **Gate 1 — Price** | Underlying stock price | $30–$500 (ETFs: no upper cap) | Penny stocks and ultra-high-priced names have thin option markets |
| **Gate 2 — Bid/Ask** | (Ask − Bid) / Mid | ≤ 10% of mid (equities), ≤ $0.05 (ETFs) | Wide spreads = immediate loss on entry; you can't fill at mid |
| **Gate 3 — Open Interest** | Contracts outstanding at strike | ≥ 1,000 | Low OI = can't exit without moving the market |
| **Gate 4 — Yield-to-Risk** | Credit / Spread Width | ≥ 20% (credit spreads only) | Below 20% is mathematically unattractive vs. margin required |
| **Gate 5 — Debit Quality** | Net Debit / Spread Width | < 75% (debit spreads only) | If you pay >75% of max profit up front, risk/reward is poor |

---

## 8. Glossary

| Term | Definition |
|------|-----------|
| **ATM** | At-The-Money — an option whose strike is equal to (or very close to) the current stock price |
| **OTM** | Out-of-The-Money — a call above the stock price, or a put below it; has no intrinsic value |
| **ITM** | In-The-Money — a call below the stock price, or a put above it; has intrinsic value |
| **Delta** | Rate of change of option price relative to a $1 move in the stock. Ranges 0–1 for calls, 0 to −1 for puts. Also interpreted as approximate probability of expiring ITM |
| **DTE** | Days To Expiration — calendar days until contract expires |
| **IV** | Implied Volatility — the market's forecast of future volatility baked into option prices |
| **IV Rank** | (Current IV − 52-wk Min) / (52-wk Max − 52-wk Min) × 100. Measures how elevated today's IV is vs. its own history |
| **HV** | Historical Volatility — annualized standard deviation of past log-returns |
| **EMA** | Exponential Moving Average — a weighted moving average that gives more weight to recent prices |
| **RSI** | Relative Strength Index — momentum oscillator (0–100). >70 = overbought, <30 = oversold |
| **Bollinger Bands** | Price envelope at 20-day SMA ± 2 standard deviations. Touches signal over-extension |
| **Credit** | Premium collected upfront when selling an options spread. Max profit if both legs expire worthless |
| **Net Debit** | Premium paid upfront when buying a spread. Maximum possible loss |
| **Spread Width** | Dollar distance between the two strikes in a vertical spread |
| **Theta** | Daily time decay. Options lose value every day. Credit sellers benefit; debit buyers are hurt |
| **Vega** | Sensitivity of option price to changes in IV. Calendars are long-vega (benefit from rising IV) |
| **Open Interest** | Total number of outstanding contracts at a given strike. Higher = more liquid |
| **Standard Monthly** | The standard options expiration: 3rd Friday of each month (or Thursday if Friday is a holiday) |
| **Credit Spread** | Two-leg options position collecting net premium: sell one option, buy another to cap risk |
| **Debit Spread** | Two-leg options position paying net premium: buy one option, sell another to reduce cost |
| **Iron Condor** | Call Credit Spread + Put Credit Spread on the same stock simultaneously |
| **Calendar Spread** | Buy and sell the same strike at different expirations; profits from time decay differential |

---

*OTIS v2.0 | Built with Python 3.12, Streamlit, yfinance | Educational use only — not financial advice*
