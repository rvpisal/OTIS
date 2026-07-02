import logging
import pickle
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from backtester import backtest_ticker
from zebra_screener import (
    evaluate_zebra,
    fetch_zebra_chain,
    ZEBRA_DTE_MIN,
    ZEBRA_DTE_MAX,
    ZEBRA_MIN_OI,
    ZEBRA_MAX_SPREAD_ABS,
    ZEBRA_EXTRINSIC_TOL,
)
from data_fetcher import DataFetcher, TICKER_UNIVERSE
from indicators import analyze_ticker, check_phase1_trigger, STRATEGY_TRIGGERS
from news_events import (
    fetch_earnings_for_tickers,
    fetch_market_headlines,
    fetch_next_earnings,
    fetch_ticker_news,
    upcoming_events,
)
from strategy_matrix import (
    screen_triggered_tickers,
    STRATEGY_META,
    CREDIT_PCT_STRATEGIES,
    DEBIT_STRATEGIES,
)

# Disk cache — survives tab close, back-navigation, and browser refreshes
CACHE_FILE = Path(__file__).parent / ".otis_results.pkl"

# Custom watchlist — persists the user's edited ticker list across restarts
WATCHLIST_FILE = Path(__file__).parent / ".otis_watchlist.txt"


def parse_watchlist(text: str) -> list[str]:
    """Parse a comma/space/newline separated ticker list.
    Uppercases, validates symbol format, dedupes while preserving order."""
    tokens = re.split(r"[,;\s]+", text.upper())
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        t = t.strip()
        if not t or not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", t):
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def load_watchlist_text() -> str:
    """Saved custom watchlist if one exists, else the default universe."""
    if WATCHLIST_FILE.exists():
        try:
            text = WATCHLIST_FILE.read_text().strip()
            if text:
                return text
        except OSError:
            pass
    return ", ".join(TICKER_UNIVERSE)


def save_watchlist(tickers: list[str]) -> None:
    """Persist a custom watchlist; remove the file when back at the default."""
    try:
        if tickers == list(TICKER_UNIVERSE):
            WATCHLIST_FILE.unlink(missing_ok=True)
        else:
            WATCHLIST_FILE.write_text(", ".join(tickers))
    except OSError as e:
        logger.warning(f"Could not save watchlist: {e}")


def save_cache(state: dict) -> None:
    """Persist pipeline results to disk so they survive browser navigation."""
    try:
        payload = {
            "results_df": state["results_df"],
            "signals_data": state["signals_data"],
            "triggered_tickers": state["triggered_tickers"],
            "total_tickers": state["total_tickers"],
            "last_run": state.get("last_run", ""),
        }
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(payload, f)
    except Exception as e:
        logger.warning(f"Could not save results cache: {e}")


def load_cache() -> dict | None:
    """Load cached pipeline results from disk. Returns None if unavailable."""
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE, "rb") as f:
            payload = pickle.load(f)
        payload["from_cache"] = True
        return payload
    except Exception as e:
        logger.warning(f"Could not load results cache: {e}")
        return None

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="OTIS — Options Trading Intelligence System",
    page_icon="📊",
    layout="wide",
)

# ── Cached external-data wrappers ─────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)        # 15 min
def cached_headlines() -> list[dict]:
    return fetch_market_headlines()


@st.cache_data(ttl=3600, show_spinner=False)       # 1 h
def cached_ticker_news(ticker: str) -> list[dict]:
    return fetch_ticker_news(ticker)


@st.cache_data(ttl=3600, show_spinner=False)       # 1 h
def cached_earnings(ticker: str):
    return fetch_next_earnings(ticker)


@st.cache_data(ttl=3600, show_spinner=False)       # 1 h — chain data changes intraday
def cached_zebra(
    ticker: str,
    dte_min: int,
    dte_max: int,
    min_oi: int,
    max_spread: float,
    extrinsic_tol: float,
    earnings_warn: bool,
    div_warn: bool,
) -> list[dict]:
    """Fetch zebra chain + evaluate for one ticker. Keyed by all filter params."""
    chain = fetch_zebra_chain(ticker, dte_min=dte_min, dte_max=dte_max)
    if not chain:
        return []
    import yfinance as yf
    hist = yf.Ticker(ticker).history(period="2y")
    price = float(hist["Close"].iloc[-1]) if hist is not None and not hist.empty else None
    if price is None:
        return []
    signals = analyze_ticker(ticker, hist)
    earnings = fetch_next_earnings(ticker)
    return evaluate_zebra(
        ticker, chain, price,
        signals=signals,
        earnings_date=earnings,
        min_oi=min_oi,
        max_spread=max_spread,
        extrinsic_tol=extrinsic_tol,
        earnings_warn=earnings_warn,
        div_warn=div_warn,
    )


@st.cache_data(ttl=86400, show_spinner=False)      # 1 day — one max-history fetch per ticker
def cached_backtest(ticker: str) -> dict | None:
    import yfinance as yf
    df = yf.Ticker(ticker).history(period="max")
    if df is None or df.empty:
        return None
    return backtest_ticker(df)

# ── Help text strings (centralised so they're easy to update) ─────────────────

_H = {
    # Sidebar
    "min_iv_rank": (
        "IV Rank (0–100) measures where today's implied volatility sits relative "
        "to the stock's own 52-week range. "
        "0 = cheapest options of the year · 100 = most expensive. "
        "Set this to 45+ to ensure you're selling premium in elevated-volatility "
        "environments, where option prices are richest."
    ),
    "min_credit_pct": (
        "The minimum net credit you must collect as a % of the spread width. "
        "Example: a 20 % minimum on a $5-wide spread means you need ≥ $1.00 in "
        "premium per share ($100 per contract). "
        "Lower values = more trades shown; higher values = only the most "
        "premium-rich setups survive."
    ),
    "dte": (
        "Days To Expiration — how many calendar days until the option contract expires. "
        "30–45 DTE is the standard credit-spread sweet spot: "
        "enough time value to collect meaningful premium, "
        "but close enough that theta (daily time decay) accelerates. "
        "Calendars use 15–30 DTE for the short leg."
    ),
    "strategies": (
        "Filter which strategy types appear in the results table.\n"
        "CREDIT (you collect premium upfront): Call/Put Credit Spreads, Iron Condor, "
        "Iron Butterfly, Cash-Secured Put, Covered Call, Short Strangle.\n"
        "DEBIT (you pay premium upfront): Bull Call & Bear Put Debit Spreads, "
        "Call Diagonal (PMCC), Put Diagonal, Call/Put Calendars, "
        "Long Call, Long Put, Long Straddle, Long Strangle.\n"
        "Each strategy's tab shows its full trigger conditions and risk profile."
    ),
    "risk_free_rate": (
        "The annualized risk-free interest rate used in the Black-Scholes model "
        "to calculate each option's delta (probability of expiring in-the-money). "
        "Set this to the current US Treasury 3-month yield. "
        "As of mid-2026 this is roughly 5 %. "
        "Changing it has a small effect on delta values and therefore which "
        "strikes are selected as short legs."
    ),
    # Metrics row
    "tickers_screened": (
        "Total tickers that passed Phase 1: had at least 200 days of price "
        "history and a closing price above the $30 floor. "
        "There is no upper price cap — option-market quality is measured "
        "directly by the per-contract liquidity gates."
    ),
    "triggered": (
        "Tickers whose technical signals matched at least one strategy trigger "
        "(e.g. RSI overbought, Bollinger Band touch, IV Rank threshold). "
        "Only these tickers had their live options chains fetched — "
        "the two-phase design avoids unnecessary API calls for the rest."
    ),
    "qualifying_setups": (
        "Setups that survived all four liquidity gates: "
        "(1) delta in the 0.15–0.20 target range, "
        "(2) bid/ask spread ≤ 10 % of mid-price (ETFs: ≤ $0.05 flat), "
        "(3) open interest ≥ 1,000 contracts, "
        "(4) credit ≥ 20 % of spread width."
    ),
    "best_credit": (
        "Highest credit-to-width ratio among all qualifying credit spread setups "
        "after filters are applied. "
        "A higher % means more premium collected relative to the maximum risk taken."
    ),
    # Signal detail
    "macro_trend": (
        "BULLISH when EMA-50 is above EMA-200 (golden cross). "
        "BEARISH when EMA-50 is below (death cross). "
        "NEUTRAL when both are within 0.5 % of each other. "
        "This is the primary directional gate: bearish/neutral stocks "
        "qualify for Call Credit Spreads; bullish stocks for Put Credit Spreads."
    ),
    "ema50": (
        "50-day Exponential Moving Average of closing prices. "
        "Responds faster to recent price action than EMA-200. "
        "When EMA-50 crosses above EMA-200 it signals the start of a bull trend "
        "(golden cross); crossing below signals a bear trend (death cross)."
    ),
    "ema200": (
        "200-day Exponential Moving Average — the long-term trend benchmark "
        "used by institutional traders worldwide. "
        "Price consistently above this line = structurally healthy bull market."
    ),
    "rsi9": (
        "Relative Strength Index over 9 periods. Measures momentum on a 0–100 scale. "
        "Above 70 = overbought (buying pressure likely exhausted, reversal risk). "
        "Below 30 = oversold (selling pressure likely exhausted, bounce likely). "
        "OTIS uses the faster 9-period version for quicker mean-reversion signals. "
        "Trigger thresholds: >72 for Call Credit Spread Group B · <28 for Put Credit Spread."
    ),
    "rsi14": (
        "Standard 14-period RSI. Slower and smoother than RSI(9). "
        "Shown for confirmation — if both RSI(9) and RSI(14) agree on direction, "
        "the signal is stronger."
    ),
    "iv_rank_detail": (
        "IV Rank = (Current IV − 52-week Low IV) / (52-week High IV − 52-week Low IV) × 100. "
        "Current IV is taken from the live ATM option price. "
        "Min/Max are derived from the stock's own 21-day Historical Volatility rolling series. "
        "Threshold: ≥ 45 = High Volatility Environment (good for selling premium). "
        "15–50 = ideal range for calendar spreads."
    ),
    "high_vol_env": (
        "True when IV Rank > 45. "
        "In a high-volatility environment, option premiums are elevated, "
        "making credit spreads more attractive — you collect more premium "
        "for the same amount of risk."
    ),
    "bb_upper": (
        "Upper Bollinger Band = 20-day SMA + 2 standard deviations. "
        "When price touches or closes above this line it is statistically "
        "stretched to the upside — a bearish mean-reversion signal. "
        "OTIS uses this as part of the Call Credit Spread Group B trigger."
    ),
    "bb_lower": (
        "Lower Bollinger Band = 20-day SMA − 2 standard deviations. "
        "When price touches or closes below this line it is statistically "
        "stretched to the downside — a bullish mean-reversion signal. "
        "OTIS uses this as part of the Put Credit Spread trigger."
    ),
}

# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    with st.sidebar:
        st.title("⚙️ Filters")

        # ── Watchlist editor ──────────────────────────────────────────────────
        # Handle a pending reset BEFORE the text_area widget is created —
        # Streamlit forbids modifying a widget's session_state after rendering.
        if st.session_state.pop("watchlist_reset", False):
            st.session_state["watchlist_text"] = ", ".join(TICKER_UNIVERSE)

        with st.expander("📋 Watchlist", expanded=False):
            wl_text = st.text_area(
                "Tickers to screen",
                value=load_watchlist_text(),
                height=160,
                key="watchlist_text",
                help=(
                    "The tickers Phase 1 scans. Separate with commas, spaces, or "
                    "new lines — case doesn't matter, duplicates are removed. "
                    "Your custom list is saved to disk when you Run Screen and "
                    "restored next time. "
                    "Note: each ticker adds ~1–3 s to Phase 1 (free-data rate "
                    "limiting), so 200 tickers ≈ 7–10 min. Per-contract liquidity "
                    "gates still apply — illiquid names are filtered out anyway."
                ),
            )
            tickers = parse_watchlist(wl_text)
            if tickers:
                is_default = tickers == list(TICKER_UNIVERSE)
                st.caption(
                    f"**{len(tickers)}** tickers parsed"
                    + (" · default list" if is_default else " · custom list")
                )
            else:
                tickers = list(TICKER_UNIVERSE)
                st.warning(
                    f"No valid tickers found — falling back to the default "
                    f"{len(TICKER_UNIVERSE)}-ticker list."
                )
            if st.button(
                "↩️ Reset to Default List", use_container_width=True,
                help="Restore the built-in Goldilocks watchlist and delete the saved custom list.",
            ):
                WATCHLIST_FILE.unlink(missing_ok=True)
                st.session_state["watchlist_reset"] = True
                st.rerun()

        min_iv_rank = st.slider(
            "Min IV Rank", 0, 100, 45, step=5,
            help=_H["min_iv_rank"],
        )
        min_credit_pct_pct = st.slider(
            "Min Credit %", 10, 50, 20, step=5,
            help=_H["min_credit_pct"],
        )
        col1, col2 = st.columns(2)
        dte_min = int(col1.number_input(
            "DTE Min", 1, 60, 30,
            help=_H["dte"],
        ))
        dte_max = int(col2.number_input(
            "DTE Max", 1, 60, 45,
            help=_H["dte"],
        ))
        all_strategies = list(STRATEGY_META.keys())
        strategies = st.multiselect(
            "Strategies",
            all_strategies,
            default=all_strategies,
            format_func=lambda k: f"{STRATEGY_META[k]['emoji']} {STRATEGY_META[k]['name']}",
            help=_H["strategies"],
        )
        risk_free_rate = st.slider(
            "Risk-Free Rate %", 0.0, 10.0, 5.0, step=0.25,
            help=_H["risk_free_rate"],
        ) / 100.0

        st.divider()

        with st.expander("📖 Quick Glossary"):
            st.markdown("""
**IV Rank** — Where current implied volatility sits in its 52-week range (0–100).
High IV Rank → option premiums are elevated → better to sell.

**Delta** — Probability (0–1) that an option expires in-the-money.
OTIS targets a short delta of 0.15–0.20, meaning ~15–20% chance of loss.

**DTE** — Days To Expiration. How long until the contract expires.

**Credit** — Premium collected when opening a credit spread. Your max profit.

**Net Debit** — Premium paid to open a calendar spread. Your max loss.

**Spread Width** — Distance between short and long strikes in a vertical spread.

**Max Loss** — Worst-case loss per contract in dollars.
Credit spread: (Width − Credit) × 100.
Calendar: Net Debit × 100.

**Open Interest (OI)** — Number of open contracts at a strike.
Higher OI = more liquidity = easier to fill and exit.

**Bollinger Bands** — Price envelope set at ±2 standard deviations from a 20-day average.
Price touching the outer band signals statistical over-extension.

**Theta** — Daily time decay. Options lose value every day they age.
Credit spread sellers benefit from theta (they're short theta by holding sold options).
""")

        st.divider()

        if CACHE_FILE.exists():
            if st.button("🗑️ Clear Saved Results", use_container_width=True,
                         help="Delete the cached results from disk and reset to a blank state. "
                              "You'll need to Run Screen again to get fresh data."):
                try:
                    CACHE_FILE.unlink(missing_ok=True)
                except Exception:
                    pass
                st.session_state.update({
                    "results_df": pd.DataFrame(),
                    "signals_data": {},
                    "triggered_tickers": [],
                    "total_tickers": 0,
                    "from_cache": False,
                    "last_run": "",
                })
                st.rerun()

        st.caption("OTIS v2.0 | Powered by yfinance")
        st.caption("⚠️ Educational use only. Not financial advice.")

    return {
        "min_iv_rank": float(min_iv_rank),
        "min_credit_pct": float(min_credit_pct_pct) / 100.0,
        "dte_min": dte_min,
        "dte_max": dte_max,
        "strategies": strategies if strategies else list(STRATEGY_META.keys()),
        "risk_free_rate": risk_free_rate,
        "tickers": tickers,
    }


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_full_pipeline(
    tickers: tuple,
    r: float,
    phase1_bar,
    phase2_bar,
    status_text,
) -> tuple[dict, dict, list, pd.DataFrame]:
    fetcher = DataFetcher(tickers=list(tickers))

    status_text.text("Phase 1: Fetching price history for all tickers...")

    def p1_cb(i, total, msg):
        phase1_bar.progress(i / total)
        status_text.text(f"[Phase 1 {i}/{total}] {msg}")

    phase1_data = fetcher.fetch_all_phase1(progress_callback=p1_cb)
    phase1_bar.progress(1.0)

    signals_data: dict = {}
    triggered_tickers: list = []

    for ticker, data in phase1_data.items():
        sigs = analyze_ticker(ticker, data["history"])
        signals_data[ticker] = sigs
        if check_phase1_trigger(sigs):
            triggered_tickers.append(ticker)

    status_text.text(
        f"Phase 1 done — {len(triggered_tickers)}/{len(phase1_data)} tickers triggered. "
        f"Fetching options chains..."
    )

    def p2_cb(i, total, msg):
        phase2_bar.progress(i / total if total > 0 else 1.0)
        status_text.text(f"[Phase 2 {i}/{total}] {msg}")

    phase2_options: dict = {}
    if triggered_tickers:
        phase2_options = fetcher.fetch_options_for_triggered(
            triggered_tickers, progress_callback=p2_cb
        )
    phase2_bar.progress(1.0)

    # Pass min_credit_pct=0.0 so ALL setups with positive credit are stored.
    # The sidebar Credit % slider applies the threshold live in apply_filters()
    # without requiring a re-run of the full pipeline.
    results_df = screen_triggered_tickers(
        phase1_data, phase2_options, signals_data,
        r=r, min_credit_pct=0.0,
    )

    # Earnings risk: a short spread held through an earnings report is the
    # cardinal sin of premium selling — flag every setup whose ticker reports
    # before expiry (surfaced via the Filter Status column).
    if not results_df.empty:
        status_text.text("Fetching earnings dates for triggered tickers...")
        earnings_map = fetch_earnings_for_tickers(
            sorted(results_df["ticker"].unique())
        )
        results_df["next_earnings"] = results_df["ticker"].map(earnings_map)

    status_text.text(
        f"✅ Complete — {len(results_df)} qualifying setup(s) found "
        f"across {len(triggered_tickers)} triggered ticker(s)."
    )
    return phase1_data, signals_data, triggered_tickers, results_df


# ── Single-ticker pipeline ────────────────────────────────────────────────────

def run_single_ticker_pipeline(
    ticker: str, r: float
) -> tuple[dict | None, pd.DataFrame]:
    """
    Run the full two-phase analysis on a single user-specified ticker.
    Phase 2 (options chain) is ALWAYS fetched regardless of trigger status,
    and the $30 watchlist price floor is bypassed — the user explicitly
    requested this ticker, so analyse it regardless of share price.

    Returns:
        (signals_dict, results_df)
        signals_dict is None if price history could not be fetched or parsed.
    """
    fetcher = DataFetcher(tickers=[ticker], enforce_price_gate=False)

    phase1_data = fetcher.fetch_all_phase1()
    if ticker not in phase1_data:
        return None, pd.DataFrame()

    signals = analyze_ticker(ticker, phase1_data[ticker]["history"])
    if signals is None:
        return None, pd.DataFrame()

    # Always fetch options for a manual lookup — ignore phase-1 trigger gate
    phase2_options = fetcher.fetch_options_for_triggered([ticker])

    results_df = screen_triggered_tickers(
        phase1_data, phase2_options, {ticker: signals},
        r=r, min_credit_pct=0.0,
    )
    if not results_df.empty:
        results_df["next_earnings"] = results_df["ticker"].map(
            {ticker: fetch_next_earnings(ticker)}
        )
    return signals, results_df


# ── Confidence scoring ────────────────────────────────────────────────────────

def compute_confidence(row: pd.Series, signals: dict) -> tuple[int, str, list[str]]:
    """
    Score a single trade setup 0–100 based on how well the technical signals
    align with and confirm the chosen strategy.

    Returns (score, label, contributing_factors_list).

    Scoring components:
      1. Trend alignment       — up to 25 pts
      2. RSI signal strength   — up to 20 pts
      3. Volatility env (IV)   — up to 20 pts
      4. Bollinger Band touch  — up to 15 pts
      5. EMA separation        — up to 10 pts
      6. Trade quality (OI/Δ%) — up to 10 pts
    """
    score = 0
    factors: list[str] = []
    strategy = str(row.get("strategy", ""))
    meta = STRATEGY_META.get(strategy, {})
    bias = meta.get("bias", "neutral")
    iv_pref = meta.get("iv_pref", "high")

    mt = str(signals.get("macro_trend", "NEUTRAL"))
    rsi9 = float(signals.get("rsi9", 50) or 50)
    iv_rank = float(signals.get("iv_rank", 50) or 50)
    bb_upper = bool(signals.get("bb_upper_touch", False))
    bb_lower = bool(signals.get("bb_lower_touch", False))
    ema50 = float(signals.get("ema50", 0) or 0)
    ema200 = float(signals.get("ema200", 1) or 1)

    # 1. Trend alignment (0–25 pts) — driven by the strategy's directional bias
    if bias == "bullish":
        if mt == "BULLISH":
            score += 25; factors.append("Bullish trend aligns with the strategy's directional bias")
        elif mt == "NEUTRAL":
            score += 10; factors.append("Neutral trend — partial alignment with bullish bias")
    elif bias == "bearish":
        if mt == "BEARISH":
            score += 25; factors.append("Bearish trend aligns with the strategy's directional bias")
        elif mt == "NEUTRAL":
            score += 15; factors.append("Neutral trend (bearish bias still qualifies)")
    elif bias == "neutral":
        if mt == "NEUTRAL":
            score += 25; factors.append("Neutral trend — ideal for a range-bound strategy")
        else:
            score += 12; factors.append(f"{mt} trend (range thesis still statistically valid)")
    else:  # volatility plays are direction-agnostic
        score += 15; factors.append("Direction-agnostic volatility play — trend not required")

    # 2. RSI signal strength (0–20 pts)
    if bias == "bullish":
        if 50 <= rsi9 <= 75:
            score += 20; factors.append(f"Upward momentum: RSI(9) = {rsi9:.0f}")
        elif rsi9 < 28:
            score += 16; factors.append(f"Deeply oversold mean-reversion setup: RSI(9) = {rsi9:.0f}")
        elif 35 <= rsi9 < 50:
            score += 8; factors.append(f"RSI(9) = {rsi9:.0f} — momentum not yet confirmed")
    elif bias == "bearish":
        if 25 <= rsi9 <= 50:
            score += 20; factors.append(f"Downward momentum: RSI(9) = {rsi9:.0f}")
        elif rsi9 > 72:
            score += 16; factors.append(f"Severely overbought reversal setup: RSI(9) = {rsi9:.0f}")
        elif rsi9 <= 60:
            score += 8; factors.append(f"RSI(9) = {rsi9:.0f} — momentum not yet confirmed")
    else:  # neutral & volatility both want RSI pinned near 50
        dist = abs(rsi9 - 50)
        if dist <= 8:
            score += 20; factors.append(f"Very neutral momentum: RSI(9) = {rsi9:.0f}")
        elif dist <= 15:
            score += 10; factors.append(f"Near-neutral momentum: RSI(9) = {rsi9:.0f}")

    # 3. Volatility environment (0–20 pts) — premium sellers want high IV,
    #    premium buyers want low IV, calendars want the middle
    if iv_pref == "high":
        if iv_rank > 70:
            score += 20; factors.append(f"Very high IV Rank = {iv_rank:.0f} — rich premium to sell")
        elif iv_rank > 55:
            score += 15; factors.append(f"High IV Rank = {iv_rank:.0f}")
        elif iv_rank > 45:
            score += 10; factors.append(f"Elevated IV Rank = {iv_rank:.0f}")
        elif iv_rank > 40:
            score += 5; factors.append(f"IV Rank = {iv_rank:.0f} — premium only moderately rich")
    elif iv_pref == "low":
        if iv_rank < 20:
            score += 20; factors.append(f"Very low IV Rank = {iv_rank:.0f} — options are cheap to buy")
        elif iv_rank < 30:
            score += 14; factors.append(f"Low IV Rank = {iv_rank:.0f}")
        elif iv_rank < 40:
            score += 8; factors.append(f"Moderate IV Rank = {iv_rank:.0f}")
        elif iv_rank < 50:
            score += 4
    else:  # mid — calendars
        if 20 <= iv_rank <= 35:
            score += 20; factors.append(f"IV Rank {iv_rank:.0f} — ideal calendar entry zone")
        elif 15 <= iv_rank <= 40:
            score += 12; factors.append(f"IV Rank {iv_rank:.0f} — acceptable for calendar")

    # 4. Bollinger Band (0–15 pts)
    if bias == "bearish" and bb_upper:
        score += 15; factors.append("Price at upper Bollinger Band — statistically overextended")
    elif bias == "bullish" and bb_lower:
        score += 15; factors.append("Price at lower Bollinger Band — statistically oversold")
    elif bias in ("neutral", "volatility") and not bb_upper and not bb_lower:
        score += 10; factors.append("Price inside Bollinger Bands — coiled, range-bound price action")

    # 5. EMA separation (0–10 pts)
    if ema200 > 0:
        sep_pct = abs(ema50 - ema200) / ema200 * 100
        if bias == "bullish" and ema50 > ema200:
            if sep_pct > 5:
                score += 10; factors.append(f"Clear golden cross — EMA50 above EMA200 by {sep_pct:.1f}%")
            elif sep_pct > 2:
                score += 6; factors.append(f"Golden cross confirmed ({sep_pct:.1f}% gap)")
        elif bias == "bearish" and ema50 < ema200:
            if sep_pct > 5:
                score += 10; factors.append(f"Clear death cross — EMA50 below EMA200 by {sep_pct:.1f}%")
            elif sep_pct > 2:
                score += 6; factors.append(f"Death cross confirmed ({sep_pct:.1f}% gap)")
        elif bias in ("neutral", "volatility"):
            if sep_pct < 1:
                score += 10; factors.append(f"EMA50/200 nearly equal ({sep_pct:.2f}% gap) — trend compression")
            elif sep_pct < 2:
                score += 5; factors.append(f"EMA50/200 converging ({sep_pct:.1f}% gap)")

    # 6. Trade quality — OI and premium ratio (0–10 pts)
    oi = int(row.get("short_oi", 0) or 0)
    credit_pct = float(row.get("credit_pct") or 0)
    debit_pct = float(row.get("debit_pct") or 0)

    if oi >= 5000:
        score += 5; factors.append(f"Excellent liquidity — OI = {oi:,}")
    elif oi >= 2000:
        score += 3; factors.append(f"Good liquidity — OI = {oi:,}")
    elif oi >= 1000:
        score += 1

    if strategy in CREDIT_PCT_STRATEGIES:
        if credit_pct >= 30:
            score += 5; factors.append(f"Strong credit = {credit_pct:.1f}% of width")
        elif credit_pct >= 25:
            score += 3; factors.append(f"Good credit = {credit_pct:.1f}% of width")
        elif credit_pct >= 20:
            score += 1
    elif debit_pct > 0:
        if debit_pct < 50:
            score += 5; factors.append(f"Attractive debit = {debit_pct:.1f}% of width/spot")
        elif debit_pct < 65:
            score += 2

    if meta.get("undefined_risk"):
        factors.append("⚠️ UNDEFINED RISK — losses unlimited beyond the short strikes; margin required")

    score = min(100, max(0, score))

    if score >= 80:   label = "⭐⭐⭐ Strong"
    elif score >= 60: label = "⭐⭐ Moderate"
    elif score >= 40: label = "⭐ Marginal"
    else:             label = "⚠️ Weak"

    return score, label, factors


def get_research_links(ticker: str) -> str:
    """Generate HTML research link bar for a ticker (opens in new tab)."""
    t = ticker.upper()
    links = [
        f'<a href="https://finance.yahoo.com/quote/{t}/chart/" target="_blank" style="text-decoration:none">📈 Yahoo Chart</a>',
        f'<a href="https://finviz.com/quote.ashx?t={t}" target="_blank" style="text-decoration:none">🔭 Finviz</a>',
        f'<a href="https://www.tradingview.com/chart/?symbol={t}" target="_blank" style="text-decoration:none">🖥 TradingView</a>',
        f'<a href="https://www.barchart.com/stocks/quotes/{t}/options" target="_blank" style="text-decoration:none">⚡ Barchart Options</a>',
    ]
    return "&nbsp;&nbsp;·&nbsp;&nbsp;".join(links)


def add_confidence_scores(df: pd.DataFrame, signals_data: dict) -> pd.DataFrame:
    """
    Add a 'confidence' column to df with format '⭐⭐ Moderate (67)'.
    Also adds 'confidence_score' (int) for numeric sorting/filtering.
    confidence_factors are NOT stored in the df — compute on demand via compute_confidence().
    """
    if df.empty:
        return df
    result = df.copy()
    conf_labels = []
    conf_scores = []
    for _, row in result.iterrows():
        sigs = signals_data.get(str(row.get("ticker", "")), {})
        if sigs:
            sc, lbl, _ = compute_confidence(row, sigs)
        else:
            sc, lbl = 0, "⚠️ Weak"
        conf_scores.append(sc)
        conf_labels.append(f"{lbl} ({sc})")
    result["confidence"] = conf_labels
    result["confidence_score"] = conf_scores
    return result


def _row_passes(row: pd.Series, filters: dict) -> bool:
    """True if a single row satisfies all current sidebar filters."""
    min_cred = filters["min_credit_pct"] * 100
    if row.get("iv_rank", 0) < filters["min_iv_rank"]:
        return False
    # Credit% (credit ÷ spread width) is only meaningful for vertical credit
    # structures — income and debit strategies skip this gate
    if row.get("strategy") in CREDIT_PCT_STRATEGIES:
        if (row.get("credit_pct") or 0) < min_cred:
            return False
    dte = row.get("dte", 0)
    if dte < filters["dte_min"] or dte > filters["dte_max"]:
        return False
    if row.get("strategy") not in filters["strategies"]:
        return False
    return True


def add_filter_flags(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Add a 'filter_status' column to every row describing which sidebar criteria
    it fails.  Rows that pass everything get '✅ Matches filters'.
    Rows that fail one or more criteria get '⚠️ <reason · reason …>'.
    """
    if df.empty:
        return df

    result = df.copy()
    min_cred = filters["min_credit_pct"] * 100

    def flags(row: pd.Series) -> str:
        issues = []
        iv = row.get("iv_rank", 0)
        if iv < filters["min_iv_rank"]:
            issues.append(f"IV Rank {iv:.0f} < {filters['min_iv_rank']:.0f}")
        if row.get("strategy") in CREDIT_PCT_STRATEGIES:
            cp = row.get("credit_pct") or 0
            if cp < min_cred:
                issues.append(f"Credit {cp:.1f}% < {min_cred:.0f}%")
        dte = row.get("dte", 0)
        if dte < filters["dte_min"] or dte > filters["dte_max"]:
            issues.append(f"DTE {dte} outside {filters['dte_min']}–{filters['dte_max']}")
        if row.get("strategy") not in filters["strategies"]:
            issues.append("strategy type hidden by filter")
        # Earnings risk: never hold a short spread through an earnings report
        ne = row.get("next_earnings")
        if ne is not None and not pd.isna(ne):
            try:
                exp_d = datetime.strptime(str(row.get("expiration", "")), "%Y-%m-%d").date()
                ne_d = ne.date() if isinstance(ne, datetime) else ne
                if isinstance(ne_d, date) and ne_d <= exp_d:
                    issues.append(f"earnings {ne_d.strftime('%b %d')} before expiry")
            except (ValueError, TypeError):
                pass
        return "✅ Matches filters" if not issues else "⚠️ " + " · ".join(issues)

    result["filter_status"] = result.apply(flags, axis=1)
    return result


def count_matching(df: pd.DataFrame, filters: dict) -> int:
    """Return how many rows pass all current sidebar filters."""
    if df.empty:
        return 0
    return int(df.apply(lambda r: _row_passes(r, filters), axis=1).sum())


# ── Leg descriptions ──────────────────────────────────────────────────────────
# The flat table schema (short_strike / long_strike / expiration) means
# different things per strategy — a calendar has two expirations, an iron
# butterfly has four strikes, a strangle's two strikes are both the same
# direction. This column spells out every leg unambiguously.

def _fmt_exp(exp) -> str:
    try:
        return datetime.strptime(str(exp), "%Y-%m-%d").strftime("%b %d")
    except (ValueError, TypeError):
        return str(exp)


def _fmt_k(k) -> str:
    try:
        if k is None or pd.isna(k):
            return "?"
        return f"{float(k):g}"
    except (TypeError, ValueError):
        return "?"


def format_legs(row: pd.Series) -> str:
    """Human-readable leg description: action · strike+type · expiry per leg."""
    s = str(row.get("strategy", ""))
    exp = _fmt_exp(row.get("expiration", ""))
    dte = row.get("dte", "?")
    ss, ls = row.get("short_strike"), row.get("long_strike")

    if s == "CALL_CREDIT_SPREAD":
        return f"Sell {_fmt_k(ss)}C + Buy {_fmt_k(ls)}C · {exp} ({dte}d)"
    if s == "PUT_CREDIT_SPREAD":
        return f"Sell {_fmt_k(ss)}P + Buy {_fmt_k(ls)}P · {exp} ({dte}d)"
    if s == "IRON_CONDOR":
        return (
            f"Sell {_fmt_k(row.get('put_short_strike'))}P + {_fmt_k(row.get('call_short_strike'))}C · "
            f"Buy {_fmt_k(row.get('put_long_strike'))}P + {_fmt_k(row.get('call_long_strike'))}C · "
            f"{exp} ({dte}d)"
        )
    if s == "IRON_BUTTERFLY":
        return (
            f"Sell {_fmt_k(ss)}C + {_fmt_k(ss)}P (body) · "
            f"Buy {_fmt_k(row.get('put_wing_strike'))}P + {_fmt_k(row.get('call_wing_strike'))}C (wings) · "
            f"{exp} ({dte}d)"
        )
    if s == "CASH_SECURED_PUT":
        return f"Sell {_fmt_k(ss)}P · {exp} ({dte}d) · cash-secured"
    if s == "COVERED_CALL":
        return f"Sell {_fmt_k(ss)}C · {exp} ({dte}d) · against 100 shares"
    if s == "SHORT_STRANGLE":
        return f"Sell {_fmt_k(ss)}P + Sell {_fmt_k(ls)}C · {exp} ({dte}d)"
    if s in ("CALL_CALENDAR_SPREAD", "PUT_CALENDAR_SPREAD"):
        cp = "C" if s.startswith("CALL") else "P"
        f_exp = _fmt_exp(row.get("front_expiration") or row.get("expiration", ""))
        b_exp = _fmt_exp(row.get("back_expiration", ""))
        return (
            f"Sell {_fmt_k(ss)}{cp} {f_exp} ({row.get('front_dte', '?')}d) · "
            f"Buy {_fmt_k(ls)}{cp} {b_exp} ({row.get('back_dte', '?')}d)"
        )
    if s in ("CALL_DIAGONAL_SPREAD", "PUT_DIAGONAL_SPREAD"):
        cp = "C" if s.startswith("CALL") else "P"
        f_exp = _fmt_exp(row.get("front_expiration") or row.get("expiration", ""))
        b_exp = _fmt_exp(row.get("back_expiration", ""))
        return (
            f"Buy {_fmt_k(ls)}{cp} {b_exp} ({row.get('back_dte', '?')}d) · "
            f"Sell {_fmt_k(ss)}{cp} {f_exp} ({row.get('front_dte', '?')}d)"
        )
    if s == "BULL_CALL_DEBIT_SPREAD":
        return f"Buy {_fmt_k(ls)}C + Sell {_fmt_k(ss)}C · {exp} ({dte}d)"
    if s == "BEAR_PUT_DEBIT_SPREAD":
        return f"Buy {_fmt_k(ls)}P + Sell {_fmt_k(ss)}P · {exp} ({dte}d)"
    if s == "LONG_CALL":
        return f"Buy {_fmt_k(ls)}C · {exp} ({dte}d)"
    if s == "LONG_PUT":
        return f"Buy {_fmt_k(ls)}P · {exp} ({dte}d)"
    if s == "LONG_STRADDLE":
        return f"Buy {_fmt_k(ls)}C + Buy {_fmt_k(ls)}P · {exp} ({dte}d)"
    if s == "LONG_STRANGLE":
        return f"Buy {_fmt_k(ss)}P + Buy {_fmt_k(ls)}C · {exp} ({dte}d)"
    return ""


# ── Styling ───────────────────────────────────────────────────────────────────

def style_results_df(df: pd.DataFrame):
    # Per-strategy leg description replaces the ambiguous flat strike/expiry
    # columns (their meaning varied by strategy — front vs back month, body vs
    # wings, etc.). Exact per-leg quotes remain available in the result dict.
    df = df.copy()
    df["legs"] = df.apply(format_legs, axis=1)

    # filter_status and confidence go first so they're always visible
    DISPLAY_COLS = [
        "filter_status",
        "confidence",
        "ticker", "current_price", "macro_trend", "rsi9", "iv_rank",
        "strategy", "legs", "dte",
        "short_oi", "credit", "net_debit", "credit_pct", "debit_pct", "max_loss",
    ]
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    display_df = df[cols].copy()

    if "net_debit" in display_df.columns:
        display_df["net_debit"] = display_df.apply(
            lambda r: r.get("net_debit") if r["strategy"] in DEBIT_STRATEGIES else None,
            axis=1,
        )
    if "credit" in display_df.columns:
        display_df["credit"] = display_df.apply(
            lambda r: None if r["strategy"] in DEBIT_STRATEGIES else r.get("credit"),
            axis=1,
        )
    if "credit_pct" in display_df.columns:
        display_df["credit_pct"] = display_df.apply(
            lambda r: None if r["strategy"] in DEBIT_STRATEGIES else r.get("credit_pct"),
            axis=1,
        )
    if "debit_pct" in display_df.columns:
        display_df["debit_pct"] = display_df.apply(
            lambda r: r.get("debit_pct") if r["strategy"] in DEBIT_STRATEGIES else None,
            axis=1,
        )

    def row_bg(row):
        # Rows that fail one or more filters get a muted grey background.
        # Every styled row pairs its light background with an explicit dark
        # text colour — otherwise Streamlit's dark theme leaves the text white
        # on a near-white background (unreadable).
        status = str(row.get("filter_status", ""))
        if status.startswith("⚠️"):
            return ["background-color: #e3e3e3; color: #555555"] * len(row)
        # Passing rows get the strategy colour from the registry
        color = STRATEGY_META.get(str(row.get("strategy", "")), {}).get("color", "#f5f5f5")
        return [f"background-color: {color}; color: #1a1a1a"] * len(row)

    def filter_status_cell(val):
        v = str(val)
        if v.startswith("✅"):
            return "background-color: #c3e6cb; color: #155724; font-weight: bold"
        if v.startswith("⚠️"):
            return "background-color: #ffeeba; color: #856404; font-style: italic"
        return ""

    def confidence_cell(val):
        v = str(val)
        if "Strong" in v:
            return "background-color: #155724; color: #fff; font-weight: bold"
        if "Moderate" in v:
            return "background-color: #1e7e34; color: #fff"
        if "Marginal" in v:
            return "background-color: #856404; color: #fff"
        if "Weak" in v:
            return "background-color: #721c24; color: #fff"
        return ""

    def oi_cell(val):
        try:
            if 1_000 <= int(val) < 2_000:
                return "background-color: #ffe0b2; color: #000"
        except (TypeError, ValueError):
            pass
        return ""

    styler = display_df.style.apply(row_bg, axis=1)
    if "filter_status" in display_df.columns:
        styler = styler.map(filter_status_cell, subset=["filter_status"])
    if "confidence" in display_df.columns:
        styler = styler.map(confidence_cell, subset=["confidence"])
    if "short_oi" in display_df.columns:
        styler = styler.map(oi_cell, subset=["short_oi"])

    fmt = {
        "current_price": "${:.2f}",
        "credit": "${:.2f}",
        "net_debit": "${:.2f}",
        "credit_pct": "{:.1f}%",
        "debit_pct": "{:.1f}%",
        "max_loss": "${:.2f}",
        "short_strike": "{:.1f}",
        "long_strike": "{:.1f}",
        "iv_rank": "{:.1f}",
        "rsi9": "{:.1f}",
        "max_profit": "${:.2f}",
    }
    styler = styler.format(
        {k: v for k, v in fmt.items() if k in display_df.columns},
        na_rep="—",
    )

    col_renames = {
        "filter_status": "Filter Status",
        "confidence": "Confidence",
        "ticker": "Ticker",
        "current_price": "Price",
        "macro_trend": "Trend",
        "rsi9": "RSI(9)",
        "iv_rank": "IV Rank",
        "strategy": "Strategy",
        "legs": "Legs (Action · Strike · Expiry)",
        "dte": "DTE",
        "short_oi": "Primary OI",
        "credit": "Credit",
        "net_debit": "Net Debit",
        "credit_pct": "Credit %",
        "debit_pct": "Debit %",
        "max_loss": "Max Loss",
    }
    styler = styler.relabel_index(
        [col_renames.get(c, c) for c in cols], axis="columns"
    )
    return styler


# ── Metrics row ───────────────────────────────────────────────────────────────

def render_metrics_row(
    results_df: pd.DataFrame,
    total_tickers: int,
    triggered_count: int,
    filters: dict,
):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tickers Screened", total_tickers, help=_H["tickers_screened"])
    c2.metric("Triggered", triggered_count, help=_H["triggered"])
    total_setups = len(results_df)
    matching = count_matching(results_df, filters)
    c3.metric(
        "Setups Found",
        total_setups,
        help=_H["qualifying_setups"],
    )
    c4.metric(
        "Match Current Filters",
        matching,
        delta=f"{total_setups - matching} outside filters" if total_setups > matching else None,
        delta_color="off",
        help=(
            "How many of the found setups pass all current sidebar filters. "
            "Non-matching setups are still shown in the table — "
            "look for the ⚠️ Filter Status column to see exactly what differs."
        ),
    )
    if not results_df.empty:
        credit_rows = results_df[results_df["strategy"].isin(CREDIT_PCT_STRATEGIES)]
        best_credit = credit_rows["credit_pct"].max() if not credit_rows.empty else 0.0
        c5.metric(
            "Best Credit %",
            f"{best_credit:.1f}%" if best_credit > 0 else "—",
            help=_H["best_credit"],
        )
    else:
        c5.metric("Best Credit %", "—", help=_H["best_credit"])


# ── Column guide (shown above the results table) ──────────────────────────────

def render_column_guide():
    with st.expander("📖 Column Guide — what does each column mean?", expanded=False):
        st.markdown("""
| Column | Meaning |
|--------|---------|
| **Filter Status** | ✅ green = passes all current sidebar filters · ⚠️ amber = shows exactly which criteria this setup falls outside of (e.g. "IV Rank 38 < 45"). Adjust the sliders to include it — no re-run needed. |
| **Ticker** | Stock or ETF symbol. |
| **Price** | The underlying's closing price **at the time of the run** (see "Data as of" timestamp). All strikes were chosen relative to this anchor — if the stock has moved since, re-run the screen. |
| **Trend** | BULLISH / BEARISH / NEUTRAL based on EMA-50 vs EMA-200. |
| **RSI(9)** | 9-period Relative Strength Index. Trigger thresholds: >72 overbought (CCS) · <28 oversold (PCS). |
| **IV Rank** | 0–100. Where today's IV sits in the stock's 52-week range. >45 = elevated premiums. |
| **Strategy** | Recommended options structure (see colour legend below). |
| **Legs** | The exact trade, leg by leg: action (Buy/Sell) · strike + type (C/P) · expiry (+ DTE). Multi-expiry strategies (calendars, diagonals) show both months; 4-leg structures (condor, butterfly) show all strikes. This is the order ticket — read it left to right. |
| **DTE** | Days To Expiration of the short / front-month leg at time of screen. All expirations are standard monthlies (3rd Friday). |
| **Short OI** | Open Interest at the primary leg's strike. Must be ≥ 1,000. **Orange** = low buffer (1,000–2,000). |
| **Credit** | Net premium **collected** per share (credit spreads only). Multiply × 100 = $ per contract. Max profit = Credit × 100. |
| **Net Debit** | Net premium **paid** per share (debit strategies & calendars). This is your maximum loss. |
| **Credit %** | Credit ÷ Spread Width × 100 (credit spreads only). Must be ≥ 20%. Higher = better risk/reward ratio. |
| **Debit %** | Net Debit ÷ Spread Width × 100 (debit spreads only). Lower = less you pay = more potential upside. Must be < 75%. |
| **Max Loss** | Worst-case dollar loss per contract. Credit spread: (Width − Credit) × 100. Debit spread / Calendar: Net Debit × 100. |

**Row colours:** 🔴 Call Credit · 🟢 Put Credit · 🟡 Iron Condor · 🔵 Call Calendar · 🟠 Bull Call Debit · 🟣 Bear Put Debit
""")


# ── Signal detail ─────────────────────────────────────────────────────────────

def render_signal_detail(ticker: str, signals_data: dict):
    sigs = signals_data.get(ticker)
    if not sigs:
        st.warning(f"No signal data available for {ticker}.")
        return

    # Research links — always shown at the top of every signal detail view
    st.markdown(
        f"**Quick Research:** &nbsp; {get_research_links(ticker)}",
        unsafe_allow_html=True,
    )
    st.caption("Links open in a new tab.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Trend & Momentum")
        st.metric("Macro Trend", sigs["macro_trend"], help=_H["macro_trend"])
        st.metric("EMA 50", f"${sigs['ema50']:.2f}", help=_H["ema50"])
        st.metric("EMA 200", f"${sigs['ema200']:.2f}", help=_H["ema200"])
        ob_label = (
            "OVERBOUGHT" if sigs["rsi9_overbought"]
            else ("OVERSOLD" if sigs["rsi9_oversold"] else None)
        )
        st.metric(
            "RSI(9)", f"{sigs['rsi9']:.1f}", delta=ob_label,
            delta_color="inverse" if sigs["rsi9_overbought"] else "normal",
            help=_H["rsi9"],
        )
        st.metric("RSI(14)", f"{sigs['rsi14']:.1f}", help=_H["rsi14"])

    with col2:
        st.subheader("Volatility & Bands")
        st.metric("Current Price", f"${sigs['current_price']:.2f}")
        st.metric("IV Rank", f"{sigs['iv_rank']:.1f}", help=_H["iv_rank_detail"])
        st.progress(
            min(sigs["iv_rank"] / 100.0, 1.0),
            text=f"IV Rank: {sigs['iv_rank']:.0f} / 100  (threshold for credit spreads: 45)",
        )
        st.metric(
            "High Vol Environment",
            "YES ✓" if sigs["high_vol_env"] else "NO ✗",
            help=_H["high_vol_env"],
        )
        st.metric(
            "Upper BB", f"${sigs['bb_upper']:.2f}",
            delta="TOUCHED" if sigs["bb_upper_touch"] else None,
            delta_color="inverse",
            help=_H["bb_upper"],
        )
        st.metric(
            "Lower BB", f"${sigs['bb_lower']:.2f}",
            delta="TOUCHED" if sigs["bb_lower_touch"] else None,
            help=_H["bb_lower"],
        )

    # ── Earnings + recent news ────────────────────────────────────────────────
    st.divider()
    next_earnings = cached_earnings(ticker)
    if next_earnings:
        days = (next_earnings - date.today()).days
        if days <= 50:
            st.warning(
                f"📅 **Next earnings: {next_earnings} ({days} days)** — earnings "
                "fall inside the typical holding window. Avoid holding short "
                "premium through the report.",
                icon="📅",
            )
        else:
            st.caption(f"📅 Next earnings: {next_earnings} ({days} days away)")
    else:
        st.caption("📅 Next earnings date unavailable.")

    news = cached_ticker_news(ticker)
    if news:
        with st.expander(f"🗞 Recent {ticker} news", expanded=False):
            for n in news:
                src = f" — _{n['source']}_" if n["source"] else ""
                if n["link"]:
                    st.markdown(f"• [{n['title']}]({n['link']}){src}")
                else:
                    st.markdown(f"• {n['title']}{src}")


# ── Strategy description banners ──────────────────────────────────────────────

_STRATEGY_INFO = {
    "CALL_CREDIT_SPREAD": (
        "🔴 **Call Credit Spread** — Sell an OTM call, buy a higher-strike call as a hedge. "
        "You collect a net credit upfront. "
        "**Profit** if the stock closes below your short strike at expiry. "
        "**Triggered by:** bearish or neutral trend · OR · overbought RSI(9) > 72 with "
        "price touching the upper Bollinger Band in an elevated-IV environment."
    ),
    "PUT_CREDIT_SPREAD": (
        "🟢 **Put Credit Spread** — Sell an OTM put, buy a lower-strike put as a hedge. "
        "You collect a net credit upfront. "
        "**Profit** if the stock closes above your short strike at expiry. "
        "**Triggered by:** bullish trend · RSI(9) < 28 · price touching the lower "
        "Bollinger Band · IV Rank > 45."
    ),
    "IRON_CONDOR": (
        "🟡 **Iron Condor** — A Call Credit Spread and Put Credit Spread on the same stock, "
        "opened simultaneously. "
        "**Profit** if the stock stays inside the range between the two short strikes at expiry. "
        "Both legs must independently qualify; the put short must be strictly below the call short."
    ),
    "CALL_CALENDAR_SPREAD": (
        "🔵 **Call Calendar Spread** — Sell a near-term ATM call (front month), buy the same "
        "strike in a later month (back month). Entered for a net debit. "
        "**Profit** when the stock stays near the strike and the front-month decays faster "
        "than the back-month. "
        "**Triggered by:** neutral-to-bullish trend · RSI(9) in 38–62 · "
        "IV Rank in 15–40 · price within 3% of EMA-50 · no Bollinger Band touch."
    ),
    "BULL_CALL_DEBIT_SPREAD": (
        "🟠 **Bull Call Debit Spread** — Buy a near-ATM call (delta ~0.50), sell a higher-strike "
        "OTM call (delta ~0.25) at the same expiry. Pay a net debit upfront. "
        "**Profit** if the stock rises above the long strike toward (or past) the short strike. "
        "Max loss = debit paid. Max profit = (spread width − debit) × 100 per contract. "
        "**Triggered by:** BULLISH trend · RSI(9) in 50–75 (upward momentum) · IV Rank < 50 "
        "(lower IV = cheaper options = smaller debit)."
    ),
    "BEAR_PUT_DEBIT_SPREAD": (
        "🟣 **Bear Put Debit Spread** — Buy a near-ATM put (delta ~−0.50), sell a lower-strike "
        "OTM put (delta ~−0.25) at the same expiry. Pay a net debit upfront. "
        "**Profit** if the stock falls below the long strike toward (or past) the short strike. "
        "Max loss = debit paid. Max profit = (spread width − debit) × 100 per contract. "
        "**Triggered by:** BEARISH or NEUTRAL trend · RSI(9) in 25–50 (downward momentum) · "
        "IV Rank < 50 (lower IV = cheaper options = smaller debit)."
    ),
    "IRON_BUTTERFLY": (
        "🦋 **Iron Butterfly** — Sell an ATM call AND an ATM put at the same strike (the body), "
        "buy protective wings 3–6% out on each side. Collects a large credit — the maximum "
        "theta harvest of any defined-risk structure. "
        "**Profit** if the stock pins the body strike at expiry; breakevens = body ± credit. "
        "**Triggered by:** neutral RSI(9) 40–60 · high-IV environment (IV Rank > 45) · "
        "no Bollinger Band touch · price within 3% of EMA-50 (pinning, not trending)."
    ),
    "CASH_SECURED_PUT": (
        "💵 **Cash-Secured Put** — Sell an OTM put (delta ~0.22) backed by enough cash to buy "
        "100 shares at the strike. "
        "**Profit:** keep the full premium if the stock stays above the strike; if assigned, "
        "you buy shares at an effective discount (strike − premium). "
        "Credit % shown = premium yield on the cash collateral. "
        "**Triggered by:** BULLISH or NEUTRAL trend · IV Rank > 45 (rich premiums)."
    ),
    "COVERED_CALL": (
        "🏦 **Covered Call** — Sell an OTM call (delta ~0.28) against 100 shares you already own. "
        "⚠️ Requires share ownership. "
        "**Profit:** keep the premium if the stock stays below the strike; if it rallies through, "
        "shares are called away at the strike (still a gain). "
        "Credit % shown = premium yield on the share cost. "
        "**Triggered by:** BULLISH or NEUTRAL trend · IV Rank > 40 · RSI(9) ≤ 70."
    ),
    "SHORT_STRANGLE": (
        "🃏 **Short Strangle** — Sell an OTM call AND an OTM put (delta ~0.15–0.20 each), same expiry. "
        "⚠️ **UNDEFINED RISK** — losses are unlimited beyond either strike; requires the highest "
        "margin approval level. Shown for completeness — most traders should prefer the Iron "
        "Condor (same thesis, defined risk). "
        "**Profit** if the stock stays between the two short strikes at expiry. "
        "**Triggered by:** neutral RSI(9) 40–60 · IV Rank > 45 · no Bollinger Band touch · "
        "price within 3% of EMA-50 (pinning, not trending)."
    ),
    "CALL_DIAGONAL_SPREAD": (
        "📐 **Call Diagonal — Poor Man's Covered Call (PMCC)** — Buy a back-month ITM call "
        "(delta ~0.70, 45–65 DTE) as a stock substitute, sell a front-month OTM call "
        "(delta ~0.25, 15–30 DTE) against it for income. "
        "Behaves like a covered call at a fraction of the capital (no 100 shares needed). "
        "**Profit** from the short leg's theta decay plus upside drift toward the short strike; "
        "the short call can be rolled month after month. "
        "Gate: net debit < strike width, so the position closes profitably even if assigned. "
        "**Triggered by:** BULLISH trend · RSI(9) in 45–70 · IV Rank < 60."
    ),
    "PUT_DIAGONAL_SPREAD": (
        "🔻 **Put Diagonal** — Buy a back-month ITM put (delta ~−0.70, 45–65 DTE), sell a "
        "front-month OTM put (delta ~−0.25, 15–30 DTE) against it. The bearish mirror of the PMCC. "
        "**Profit** from the short leg's decay plus downside drift toward the short strike. "
        "Gate: net debit < strike width (assignment-safe). "
        "**Triggered by:** BEARISH trend · RSI(9) in 30–55 · IV Rank < 60."
    ),
    "PUT_CALENDAR_SPREAD": (
        "🌊 **Put Calendar Spread** — Sell a near-term ATM put (front month), buy the same strike "
        "in a later month (back month). Entered for a net debit. "
        "**Profit** when the stock stays near the strike and the front-month decays faster than "
        "the back month. Bearish-leaning twin of the call calendar. "
        "**Triggered by:** BEARISH or NEUTRAL trend · RSI(9) in 38–62 · IV Rank in 15–40 · "
        "price within 3% of EMA-50 · no Bollinger Band touch."
    ),
    "LONG_CALL": (
        "🚀 **Long Call** — Buy a slightly-ITM call (delta ~0.65): retains intrinsic value and "
        "bleeds less theta than ATM/OTM lottery tickets. Unlimited upside; max loss = premium paid. "
        "Breakeven = strike + premium. "
        "**Triggered by:** BULLISH trend · RSI(9) in 55–75 (strong momentum) · IV Rank < 40 "
        "(don't overpay for volatility)."
    ),
    "LONG_PUT": (
        "🪂 **Long Put** — Buy a slightly-ITM put (delta ~−0.65). Profits as the stock falls; "
        "max loss = premium paid. Breakeven = strike − premium. "
        "**Triggered by:** BEARISH trend · RSI(9) in 25–45 (downward momentum) · IV Rank < 40."
    ),
    "LONG_STRADDLE": (
        "🎯 **Long Straddle** — Buy an ATM call AND an ATM put at the same strike & expiry. "
        "**Profit** on a large move in EITHER direction, or an IV expansion. "
        "Needs the stock to move more than the combined debit (shown as Debit % = breakeven "
        "move as % of spot; capped at 12%). "
        "**Triggered by:** IV Rank < 25 (cheap options) · neutral RSI(9) 40–60 · price coiled "
        "inside the Bollinger Bands."
    ),
    "LONG_STRANGLE": (
        "🎪 **Long Strangle** — Buy an OTM call AND an OTM put (delta ~0.25 each), same expiry. "
        "Cheaper than a straddle, but needs a bigger move to pay off (debit capped at 8% of spot). "
        "The two strikes shown are the put (lower) and call (upper) — both legs are LONG. "
        "**Triggered by:** IV Rank < 25 · neutral RSI(9) 40–60 · price coiled inside the bands."
    ),
}


# ── Results table ─────────────────────────────────────────────────────────────

def render_results_table(results_df: pd.DataFrame, filters: dict, signals_data: dict):
    if results_df.empty:
        st.info("No qualifying setups were found. Try running the screen again.")
        return

    # Annotate every row with filter-compliance info — no rows are hidden
    flagged_df = add_filter_flags(results_df, filters)
    flagged_df = add_confidence_scores(flagged_df, signals_data)
    matching = count_matching(results_df, filters)
    total = len(flagged_df)

    if matching == 0:
        st.warning(
            f"All {total} found setup{'s' if total != 1 else ''} are shown below. "
            "None currently pass all sidebar filters — the **Filter Status** column "
            "shows exactly what differs. Adjust the sliders to include more setups; "
            "no re-run is needed."
        )
    elif matching < total:
        st.info(
            f"**{matching} of {total} setups match your current filters** "
            f"({total - matching} shown with ⚠️). "
            "Adjust sidebar sliders to include more — changes apply instantly."
        )

    render_column_guide()

    # Dynamic tabs: "All" + one tab per strategy that actually has results —
    # with 17 strategy types, empty tabs would just be noise
    present = [k for k in STRATEGY_META if (flagged_df["strategy"] == k).any()]

    # Tab labels show (matching / total) so the user can see filter impact per strategy
    def tab_label(name: str, tab_df: pd.DataFrame) -> str:
        m = int(tab_df["filter_status"].str.startswith("✅").sum()) if not tab_df.empty else 0
        return f"{name} ({m}/{len(tab_df)})"

    labels = [tab_label("All", flagged_df)] + [
        tab_label(
            f"{STRATEGY_META[k]['emoji']} {STRATEGY_META[k]['name']}",
            flagged_df[flagged_df["strategy"] == k],
        )
        for k in present
    ]
    tabs = st.tabs(labels)

    for tab, strategy_key in zip(tabs, [None] + present):
        tab_df = (
            flagged_df if strategy_key is None
            else flagged_df[flagged_df["strategy"] == strategy_key]
        )
        key_suffix = (strategy_key or "all").lower()
        with tab:
            if strategy_key and strategy_key in _STRATEGY_INFO:
                st.info(_STRATEGY_INFO[strategy_key])
            elif strategy_key is None:
                st.caption(
                    "Showing all strategies — switch to a strategy tab for setup details. "
                    "Coloured row = passes current filters · ⚠️ grey row = outside filter range "
                    "(see Filter Status column). Row colour identifies the strategy type."
                )

            st.dataframe(
                style_results_df(tab_df),
                use_container_width=True,
                hide_index=True,
            )

            tickers_in_tab = ["— select ticker —"] + list(tab_df["ticker"].unique())
            selected = st.selectbox(
                "View signal detail for:",
                options=tickers_in_tab,
                key=f"sel_{key_suffix}",
                help="Select a ticker to see the full technical indicator breakdown that triggered this setup.",
            )
            if selected != "— select ticker —":
                with st.expander(f"📋 Signal Detail: {selected}", expanded=True):
                    render_signal_detail(selected, signals_data)


# ── Single ticker lookup UI ───────────────────────────────────────────────────

def render_single_ticker_section(filters: dict):
    """
    UI section: analyze any single ticker on demand (full phase 1 + phase 2).
    Results are stored in session_state so they survive filter changes.
    """
    col_input, col_btn, col_clear = st.columns([3, 1, 1])
    ticker_raw = col_input.text_input(
        "Ticker symbol",
        placeholder="e.g. AAPL, TSLA, SPY, NVDA …",
        label_visibility="collapsed",
        key="single_lookup_input",
        help=(
            "Enter any US equity or ETF symbol. "
            "Runs the full two-phase analysis (price history → technicals → "
            "live options chain → all 17 strategy evaluations) on this one ticker. "
            "Not limited to the Goldilocks watchlist."
        ),
    ).strip().upper()

    analyze_clicked = col_btn.button(
        "🔍 Analyze", type="primary", use_container_width=True,
        key="single_lookup_btn",
        help="Fetch fresh market data and run all 17 strategy evaluations for this ticker.",
    )
    clear_clicked = col_clear.button(
        "✕ Clear", use_container_width=True,
        key="single_lookup_clear",
        help="Clear single-ticker results.",
    )

    if clear_clicked:
        for k in (
            "single_lookup_result_signals",
            "single_lookup_result_df",
            "single_lookup_result_ticker",
            "single_lookup_error",
        ):
            st.session_state.pop(k, None)
        st.rerun()

    if analyze_clicked and ticker_raw:
        st.session_state.pop("single_lookup_error", None)
        with st.spinner(f"Fetching data for **{ticker_raw}** — price history + live options chain…"):
            try:
                signals, results_df = run_single_ticker_pipeline(
                    ticker_raw, filters["risk_free_rate"]
                )
                if signals is None:
                    st.session_state["single_lookup_error"] = (
                        f"**{ticker_raw}** could not be analysed. Possible reasons: "
                        "the ticker was not found on Yahoo Finance, it has under "
                        "200 trading days of history (recent IPOs), or a network error. "
                        "Double-check the symbol and try again."
                    )
                    # clear stale results
                    st.session_state.pop("single_lookup_result_ticker", None)
                    st.session_state.pop("single_lookup_result_signals", None)
                    st.session_state.pop("single_lookup_result_df", None)
                else:
                    st.session_state["single_lookup_result_ticker"] = ticker_raw
                    st.session_state["single_lookup_result_signals"] = signals
                    st.session_state["single_lookup_result_df"] = results_df
                    st.session_state.pop("single_lookup_error", None)
            except Exception as exc:
                st.session_state["single_lookup_error"] = (
                    f"Pipeline error for **{ticker_raw}**: {exc}"
                )

    # ── Display cached lookup results ─────────────────────────────────────────
    if err := st.session_state.get("single_lookup_error"):
        st.error(err)
        return

    result_ticker  = st.session_state.get("single_lookup_result_ticker")
    result_signals = st.session_state.get("single_lookup_result_signals")
    result_df      = st.session_state.get("single_lookup_result_df")

    if result_ticker is None or result_signals is None:
        return  # no results yet — clean state

    st.divider()

    # ── Technical signals ──────────────────────────────────────────────────────
    with st.expander(
        f"📋 Technical Signals: {result_ticker}", expanded=True
    ):
        render_signal_detail(result_ticker, {result_ticker: result_signals})

    # ── Strategy results ───────────────────────────────────────────────────────
    if result_df is None or result_df.empty:
        triggered = check_phase1_trigger(result_signals)
        if triggered:
            st.warning(
                f"⚠️ Technical signals triggered for **{result_ticker}**, but no setup "
                "passed all liquidity gates (delta 0.15–0.20 range, bid/ask ≤ 10%, "
                "OI ≥ 1,000, yield ≥ 20% of width). "
                "The options chain may lack sufficient liquidity at qualifying strikes today."
            )
        else:
            st.info(
                f"ℹ️ No strategy triggers fired for **{result_ticker}** under current conditions. "
                "Check the signal panel above — the combination of trend, RSI, IV Rank, "
                "and Bollinger Bands does not yet match any of the 17 strategy templates."
            )
        return

    # Annotate with filter flags and confidence
    flagged = add_filter_flags(result_df, filters)
    flagged = add_confidence_scores(flagged, {result_ticker: result_signals})

    n       = len(flagged)
    n_match = int(flagged["filter_status"].str.startswith("✅").sum())
    st.success(
        f"Found **{n}** qualifying setup{'s' if n != 1 else ''} for **{result_ticker}** "
        f"({n_match} match{'es' if n_match == 1 else ''} current sidebar filters)."
    )
    st.dataframe(style_results_df(flagged), use_container_width=True, hide_index=True)

    # ── Confidence breakdown (expandable per setup) ────────────────────────────
    st.subheader("📊 Confidence Breakdown")
    st.caption(
        "Each setup is scored 0–100 based on signal alignment, RSI strength, "
        "IV environment, Bollinger Band touch, EMA separation, and trade quality."
    )
    for _, row in flagged.iterrows():
        score    = int(row.get("confidence_score", 0))
        conf_str = str(row.get("confidence", ""))
        strat    = str(row.get("strategy", ""))
        _, _, factors = compute_confidence(row, result_signals)
        with st.expander(f"{strat} — {conf_str}", expanded=(score >= 60)):
            if factors:
                for f in factors:
                    st.markdown(f"&nbsp;&nbsp;✔ {f}")
            else:
                st.caption("No specific confidence factors identified for this setup.")


# ── News & Events ─────────────────────────────────────────────────────────────

def render_news_section():
    with st.expander("📰 News & Events", expanded=False):
        tab_macro, tab_news = st.tabs(["📅 Macro Calendar", "🗞 Market Headlines"])

        with tab_macro:
            events = upcoming_events(days_ahead=45)
            if not events:
                st.info("No scheduled macro events in the next 45 days.")
            else:
                st.caption(
                    "Scheduled market-moving releases in the next 45 days — every one of "
                    "these lands **inside the 30–45 DTE window** of a spread opened today. "
                    "🔴 = high impact (market-wide volatility event)."
                )
                for ev in events:
                    icon = "🔴" if ev["importance"] == "high" else "🟡"
                    days = ev["days_until"]
                    when = "**TODAY**" if days == 0 else f"in **{days}** day{'s' if days != 1 else ''}"
                    line = f"{icon} **{ev['date']}** ({when}) — **{ev['event']}** · {ev['note']}"
                    if ev["importance"] == "high" and days <= 7:
                        st.error(line, icon="⚠️")
                    else:
                        st.markdown(line)
                st.caption(
                    "FOMC dates: Federal Reserve published calendar · CPI dates: BLS "
                    "published schedule · NFP: first Friday · PCE/GDP dates approximate (±1 day)."
                )

        with tab_news:
            headlines = cached_headlines()
            if not headlines:
                st.info("Could not load headlines (feeds unreachable). Try again in a minute.")
            else:
                now = datetime.now()
                for h in headlines:
                    age = ""
                    if h["published"]:
                        hrs = max(0, (now - h["published"]).total_seconds() / 3600)
                        age = f"{hrs:.0f}h ago" if hrs >= 1 else f"{hrs * 60:.0f}m ago"
                    meta = " · ".join(x for x in (h["source"], age) if x)
                    if h["link"]:
                        st.markdown(f"• [{h['title']}]({h['link']})  \n&nbsp;&nbsp;_{meta}_")
                    else:
                        st.markdown(f"• {h['title']}  \n&nbsp;&nbsp;_{meta}_")
                st.caption("Sources: Yahoo Finance & CNBC RSS · refreshes every 15 minutes.")


# ── Zebra — Stock Replacement Screener ───────────────────────────────────────

def _zebra_card(r: dict) -> None:
    """Render one Zebra trade card inside a bordered container."""
    ext_label = f"${r['net_extrinsic']:+.2f}/shr" if r["net_extrinsic"] != 0 else "$0.00/shr ✨"
    ext_color = "#1e7e34" if abs(r["net_extrinsic"]) <= 0.05 else "#a8d5b0" if abs(r["net_extrinsic"]) <= 0.10 else "#fff3cd"

    warnings = r.get("warnings", [])
    warn_html = "".join(
        f'<span style="background:#fff3cd;color:#856404;border-radius:4px;padding:2px 7px;'
        f'margin:2px;font-size:0.82em;display:inline-block">⚠️ {w}</span>'
        for w in warnings
    ) if warnings else '<span style="color:#1e7e34;font-size:0.85em">✅ No warnings</span>'

    with st.container(border=True):
        # Header
        hc1, hc2, hc3 = st.columns([2, 2, 1])
        hc1.markdown(
            f"### 🦓 {r['ticker']}  "
            f"<span style='font-size:0.85em;color:#888'>${r['current_price']:,.2f}</span>",
            unsafe_allow_html=True,
        )
        hc2.markdown(
            f"**Buy 2× ${r['long_strike']:g}C &nbsp;·&nbsp; Sell 1× ${r['short_strike']:g}C**  \n"
            f"Expiry: {r['expiration']} &nbsp;({r['dte']}d)"
        )
        hc3.markdown(
            f"<div style='background:{ext_color};border-radius:6px;padding:6px 10px;"
            f"text-align:center;font-weight:bold'>Net Extrinsic<br>{ext_label}</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # Key metrics — two rows of 4
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Net Debit", f"${r['net_debit']:.2f}/shr", help="Total cost to open (2×long – 1×short), per share equivalent")
        m2.metric("Max Loss", f"${r['max_loss']:,.0f}", help="Net debit × 100 — the most you can lose (defined risk)")
        m3.metric("Breakeven", f"${r['breakeven']:,.2f}", help="Long strike + net debit — stock must be above this at expiry to profit")
        m4.metric("Structure Δ", f"{r['structure_delta']:.2f}", help="2×long delta − short delta ≈ delta of owning 100 shares")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Capital Required", f"${r['capital_required']:,.0f}", help="Net debit × 100 — what you pay to open")
        m6.metric("vs 100 Shares", f"${r['share_cost']:,.0f}", help="Cost of outright stock ownership at current price")
        m7.metric("Capital Saved", f"{r['capital_efficiency']:.1f}%", help="How much less capital this uses vs buying 100 shares")
        m8.metric("IV Rank", f"{r['iv_rank']:.0f}", help="Low IV = better entry for this positive-vega structure")

        # Leg details
        st.markdown("**Leg details**")
        ld1, ld2 = st.columns(2)
        ld1.caption(
            f"**Long ×2 — ${r['long_strike']:g} Call**  \n"
            f"Mid: ${r['long_mid']:.2f} &nbsp;|&nbsp; "
            f"Bid/Ask: ${r['long_bid']:.2f} / ${r['long_ask']:.2f} "
            f"(spread ${r['long_spread']:.2f})  \n"
            f"Delta: {r['long_delta']:.2f} &nbsp;|&nbsp; OI: {r['long_oi']:,} &nbsp;|&nbsp; "
            f"Extrinsic: ${r['long_ext']:.2f}"
        )
        ld2.caption(
            f"**Short ×1 — ${r['short_strike']:g} Call**  \n"
            f"Mid: ${r['short_mid']:.2f} &nbsp;|&nbsp; "
            f"Bid/Ask: ${r['short_bid']:.2f} / ${r['short_ask']:.2f} "
            f"(spread ${r['short_spread']:.2f})  \n"
            f"Delta: {r['short_delta']:.2f} &nbsp;|&nbsp; OI: {r['short_oi']:,} &nbsp;|&nbsp; "
            f"Extrinsic: ${r['short_ext']:.2f}"
        )

        gran = r.get("strike_granularity", 5.0)
        gran_badge = f"✅ ${gran:g} strike increments" if gran <= 2.5 else f"⚠️ ${gran:g} strike increments (coarse)"
        st.caption(f"Strike grid: {gran_badge}")

        # Warnings
        st.markdown(warn_html, unsafe_allow_html=True)


def _zebra_filters() -> dict:
    """Render Zebra filter controls and return the active config."""
    with st.expander("⚙️ Zebra Filters", expanded=False):
        fc1, fc2, fc3, fc4 = st.columns(4)
        dte_min      = fc1.number_input("Min DTE",         min_value=30,   max_value=90,   value=ZEBRA_DTE_MIN, step=5,    key="z_dte_min")
        dte_max      = fc2.number_input("Max DTE",         min_value=60,   max_value=180,  value=ZEBRA_DTE_MAX, step=10,   key="z_dte_max")
        min_oi       = fc3.number_input("Min OI (legs)",   min_value=50,   max_value=2000, value=ZEBRA_MIN_OI,  step=50,   key="z_min_oi")
        max_spread   = fc4.number_input("Max Spread $",    min_value=0.05, max_value=0.50, value=ZEBRA_MAX_SPREAD_ABS, step=0.05, format="%.2f", key="z_spread")

        fc5, fc6, fc7 = st.columns([1, 1, 2])
        extrinsic_tol = fc5.number_input("Extrinsic Tol $", min_value=0.02, max_value=0.50, value=ZEBRA_EXTRINSIC_TOL, step=0.02, format="%.2f", key="z_ext_tol")
        earnings_warn = fc6.toggle("Earnings warning",     value=True, key="z_earn_warn")
        div_warn      = fc7.toggle("Dividend risk warning", value=True, key="z_div_warn")

    return {
        "dte_min": int(dte_min), "dte_max": int(dte_max),
        "min_oi": int(min_oi), "max_spread": float(max_spread),
        "extrinsic_tol": float(extrinsic_tol),
        "earnings_warn": bool(earnings_warn), "div_warn": bool(div_warn),
    }


def render_zebra_section(signals_data: dict) -> None:
    st.caption(
        "A **Zebra** (Zero-Extrinsic Back Ratio) mimics owning 100 shares with defined downside risk "
        "and 30–70% less capital: **Buy 2× deep ITM call (~80–85Δ) + Sell 1× ATM call (~50Δ)**, "
        "same expiry. The ratio is sized so net extrinsic ≈ $0, making it delta-equivalent to stock "
        "without unlimited downside. Best in **bullish, low-to-moderate IV** environments."
    )

    cfg = _zebra_filters()

    # ── Manual ticker input ───────────────────────────────────────────────────
    col_in, col_btn = st.columns([3, 1])
    z_ticker = col_in.text_input(
        "Analyse ticker", placeholder="e.g. AAPL, NVDA, SPY …",
        label_visibility="collapsed", key="z_ticker_input",
    ).strip().upper()
    run_single = col_btn.button("🦓 Analyse", type="primary",
                                use_container_width=True, key="z_single_btn")

    # ── Watchlist scan (bullish tickers from last screen run) ─────────────────
    bullish_tickers = sorted([
        t for t, s in signals_data.items()
        if s and s.get("macro_trend") == "BULLISH"
    ])
    if bullish_tickers:
        st.caption(
            f"**{len(bullish_tickers)} bullish tickers** from the last watchlist screen "
            f"({', '.join(bullish_tickers[:8])}{'…' if len(bullish_tickers) > 8 else ''}) "
            "are candidates for Zebra setups."
        )
        scan_btn = st.button(
            f"🔍 Scan all {len(bullish_tickers)} bullish tickers for Zebra setups",
            key="z_scan_btn",
        )
    else:
        scan_btn = False
        if signals_data:
            st.caption("No bullish tickers in the last screen — use manual lookup above or run the screen first.")

    # ── Handle single ticker ──────────────────────────────────────────────────
    if run_single and z_ticker:
        with st.spinner(f"Fetching Zebra chain for **{z_ticker}**… (60–{cfg['dte_max']}d monthly expirations)"):
            result = cached_zebra(
                z_ticker,
                cfg["dte_min"], cfg["dte_max"],
                cfg["min_oi"], cfg["max_spread"],
                cfg["extrinsic_tol"],
                cfg["earnings_warn"], cfg["div_warn"],
            )
        st.session_state["zebra_single"] = (z_ticker, result)
        st.session_state.pop("zebra_scan", None)

    # ── Handle watchlist scan ─────────────────────────────────────────────────
    if scan_btn and bullish_tickers:
        scan_results: list[dict] = []
        prog = st.progress(0, text="Starting scan…")
        for i, t in enumerate(bullish_tickers):
            prog.progress((i + 1) / len(bullish_tickers), text=f"Scanning {t} ({i+1}/{len(bullish_tickers)})…")
            rows = cached_zebra(
                t,
                cfg["dte_min"], cfg["dte_max"],
                cfg["min_oi"], cfg["max_spread"],
                cfg["extrinsic_tol"],
                cfg["earnings_warn"], cfg["div_warn"],
            )
            scan_results.extend(rows)
        prog.empty()
        # Best structure per ticker (closest to zero extrinsic)
        seen: set[str] = set()
        deduped: list[dict] = []
        for r in sorted(scan_results, key=lambda x: abs(x["net_extrinsic"])):
            if r["ticker"] not in seen:
                seen.add(r["ticker"])
                deduped.append(r)
        st.session_state["zebra_scan"] = deduped
        st.session_state.pop("zebra_single", None)

    # ── Display results ───────────────────────────────────────────────────────
    if "zebra_single" in st.session_state:
        ticker_shown, rows = st.session_state["zebra_single"]
        st.subheader(f"🦓 {ticker_shown} — Zebra Structures")
        if not rows:
            st.warning(
                f"No qualifying Zebra structures found for **{ticker_shown}** with current filters. "
                "Try relaxing Min OI, increasing Extrinsic Tolerance, or checking a different DTE range."
            )
        else:
            st.caption(f"{len(rows)} structure{'s' if len(rows) > 1 else ''} — sorted by |net extrinsic| (closest to zero first)")
            for r in rows:
                _zebra_card(r)

    elif "zebra_scan" in st.session_state:
        scan_rows = st.session_state["zebra_scan"]
        st.subheader(f"🦓 Watchlist Zebra Scan — {len(scan_rows)} tickers with valid structures")
        if not scan_rows:
            st.warning(
                "No qualifying Zebra structures found for any bullish ticker. "
                "Relax filters or wait for higher liquidity in the relevant DTE window."
            )
        else:
            # Sort by capital efficiency (best savings first)
            for r in sorted(scan_rows, key=lambda x: -x["capital_efficiency"]):
                _zebra_card(r)

    st.caption(
        "⚠️ **Structure note:** Net extrinsic ≈ $0 is the target — a slightly negative value "
        "(credit on the extrinsic balance) is acceptable; a large positive means you're paying "
        "excess time value on the longs. Structure delta ~1.1–1.2 is typical (slightly more than "
        "100 shares). Max loss is the net debit × 100 — always defined, unlike long stock."
    )


# ── Backtest ──────────────────────────────────────────────────────────────────

def _verdict(win45: float) -> str:
    """Plain-English verdict badge from 45-day win rate."""
    if pd.isna(win45):
        return "—"
    if win45 >= 65:
        return "✅ Strong"
    if win45 >= 50:
        return "⚠️ Mixed"
    return "❌ Weak"


def _build_simple_bt_df(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Collapse the raw backtest DataFrame to 5 readable columns:
      label_col | # Times Fired | Win% (3 weeks) | Win% (6 weeks) | Avg Move | Worst Case | Signal
    Only 21d and 45d horizons shown — the two that matter for standard option DTEs.
    """
    rows = []
    for idx, row in df.iterrows():
        w21 = row.get("win_21d", float("nan"))
        w45 = row.get("win_45d", float("nan"))
        avg45 = row.get("avg_45d", float("nan"))
        worst45 = row.get("worst_45d", float("nan"))
        rows.append({
            label_col: str(idx),
            "# Times Fired": int(row.get("occurrences", 0)),
            "Win% (3 wk)": f"{w21:.0f}%" if not pd.isna(w21) else "—",
            "Win% (6 wk)": f"{w45:.0f}%" if not pd.isna(w45) else "—",
            "Avg Move (6 wk)": (f"+{avg45:.1f}%" if avg45 >= 0 else f"{avg45:.1f}%") if not pd.isna(avg45) else "—",
            "Worst Case (6 wk)": f"{worst45:.1f}%" if not pd.isna(worst45) else "—",
            "Signal": _verdict(w45),
        })
    return pd.DataFrame(rows).set_index(label_col)


def _color_simple_bt(df: pd.DataFrame):
    """Color only the Signal column and Win% columns for quick scanning."""
    def signal_cell(v):
        if "Strong" in str(v):
            return "background-color: #1e7e34; color: #fff; font-weight: bold"
        if "Mixed" in str(v):
            return "background-color: #fff3cd; color: #1a1a1a; font-weight: bold"
        if "Weak" in str(v):
            return "background-color: #f8d7da; color: #1a1a1a; font-weight: bold"
        return ""

    def win_cell(v):
        try:
            n = float(str(v).replace("%", ""))
        except (TypeError, ValueError):
            return ""
        if n >= 65:
            return "background-color: #d4edda; color: #1a1a1a"
        if n >= 50:
            return "background-color: #fff3cd; color: #1a1a1a"
        return "background-color: #f8d7da; color: #1a1a1a"

    win_cols = [c for c in df.columns if c.startswith("Win%")]
    styler = df.style
    if win_cols:
        styler = styler.map(win_cell, subset=win_cols)
    if "Signal" in df.columns:
        styler = styler.map(signal_cell, subset=["Signal"])
    return styler


def _render_backtest_summary(ticker: str, live_sigs: dict | None, res: dict):
    """
    Plain-English summary: what signals are firing RIGHT NOW and what history
    says about each of them.
    """
    if not live_sigs:
        return

    # ── Current market snapshot ───────────────────────────────────────────────
    mt = str(live_sigs.get("macro_trend", "NEUTRAL"))
    rsi = float(live_sigs.get("rsi9", 50) or 50)
    iv = float(live_sigs.get("iv_rank", 50) or 50)
    price = float(live_sigs.get("current_price", 0) or 0)
    ema50 = float(live_sigs.get("ema50", 0) or 0)
    ema200 = float(live_sigs.get("ema200", 0) or 0)

    trend_desc = {"BULLISH": "📈 Bullish (EMA-50 above EMA-200)",
                  "BEARISH": "📉 Bearish (EMA-50 below EMA-200)",
                  "NEUTRAL": "➡️ Neutral (EMAs overlapping)"}[mt]
    rsi_desc = "oversold" if rsi < 30 else ("overbought" if rsi > 70 else "neutral")
    iv_desc = "low" if iv < 35 else ("elevated" if iv > 60 else "moderate")

    st.markdown("### 📊 Current Conditions + Historical Outlook")
    cols = st.columns(4)
    cols[0].metric("Price", f"${price:,.2f}")
    cols[1].metric("Trend", mt.capitalize())
    cols[2].metric("RSI(9)", f"{rsi:.1f}", delta=rsi_desc)
    cols[3].metric("IV Rank", f"{iv:.0f}", delta=iv_desc)

    st.markdown(
        f"**{ticker}** is currently in a **{mt.lower()} trend** — EMA-50 is "
        f"${ema50:,.2f} vs EMA-200 at ${ema200:,.2f}. "
        f"RSI(9) is **{rsi:.1f}** ({rsi_desc}), and implied volatility rank is "
        f"**{iv:.0f}/100** ({iv_desc})."
    )

    # ── Which triggers fire right now? ────────────────────────────────────────
    fired = [k for k, fn in STRATEGY_TRIGGERS.items() if fn(live_sigs)]

    if not fired:
        st.info(
            "No OTIS strategy triggers are active on this ticker right now. "
            "The backtest tables below show historical performance for all strategies — "
            "check back when conditions shift."
        )
        return

    # ── Look up fired strategies in backtest results ──────────────────────────
    strat_df = res.get("strategies", pd.DataFrame())
    meta_lookup = {}
    for name, row in strat_df.iterrows():
        # Match by strategy key embedded in the emoji-prefixed row label
        for k in fired:
            if k.replace("_", " ").lower() in str(name).lower():
                meta_lookup[k] = row
                break

    st.markdown("#### Active signals right now")
    fired_meta = [STRATEGY_META.get(k, {}) for k in fired]
    signal_labels = [
        f"{m.get('emoji','•')} {m.get('name', k)}" for k, m in zip(fired, fired_meta)
    ]
    st.markdown("  ".join(f"`{lbl}`" for lbl in signal_labels))

    # ── Per-fired-strategy outcome summary ────────────────────────────────────
    st.markdown("#### What history says about each signal")
    for k in fired:
        meta = STRATEGY_META.get(k, {})
        name = meta.get("name", k.replace("_", " ").title())
        emoji = meta.get("emoji", "•")
        bias = meta.get("bias", "bullish")
        row = meta_lookup.get(k)

        if row is None:
            st.markdown(f"**{emoji} {name}** — not enough historical triggers to measure.")
            continue

        occ = int(row.get("occurrences", 0))
        w21 = row.get("win_21d", float("nan"))
        w45 = row.get("win_45d", float("nan"))
        avg45 = row.get("avg_45d", float("nan"))
        worst45 = row.get("worst_45d", float("nan"))
        verdict = _verdict(w45)

        direction = {"bullish": "higher", "bearish": "lower",
                     "neutral": "within a tight range", "volatility": "significantly in either direction"}[bias]

        w21_str = f"{w21:.0f}%" if not pd.isna(w21) else "—"
        w45_str = f"{w45:.0f}%" if not pd.isna(w45) else "—"
        avg_str = (f"+{avg45:.1f}%" if avg45 >= 0 else f"{avg45:.1f}%") if not pd.isna(avg45) else "—"
        worst_str = f"{worst45:.1f}%" if not pd.isna(worst45) else "—"

        with st.container(border=True):
            st.markdown(
                f"**{emoji} {name}** &nbsp; {verdict}\n\n"
                f"This trigger has fired **{occ} times** historically. "
                f"In **{w21_str}** of those cases the stock moved {direction} within 3 weeks, "
                f"and **{w45_str}** of the time within 6 weeks — the typical options holding window. "
                f"The average 6-week price move was **{avg_str}**. "
                f"Worst single outcome: **{worst_str}**."
            )

    # ── Bottom-line outlook ───────────────────────────────────────────────────
    # Aggregate win rate across all matched fired strategies at 45d
    matched_w45 = [
        float(meta_lookup[k].get("win_45d", float("nan")))
        for k in fired if k in meta_lookup and not pd.isna(meta_lookup[k].get("win_45d", float("nan")))
    ]
    if matched_w45:
        avg_w45 = sum(matched_w45) / len(matched_w45)
        if avg_w45 >= 65:
            outlook = (
                f"📈 **Historical edge is in your favour.** "
                f"Across active signals, the stock moved in the expected direction about "
                f"**{avg_w45:.0f}%** of the time over 6 weeks. This is a statistically meaningful edge — "
                f"but watch for earnings or macro events before expiry."
            )
        elif avg_w45 >= 50:
            outlook = (
                f"⚠️ **Edge is marginal.** "
                f"Active signals have a combined 6-week win rate of **{avg_w45:.0f}%** historically — "
                f"slightly better than a coin flip. Size accordingly and keep stops tight."
            )
        else:
            outlook = (
                f"❌ **Signals have underperformed historically.** "
                f"Active signals show only a **{avg_w45:.0f}%** 6-week win rate — "
                f"be cautious. Consider waiting for a cleaner setup."
            )
        st.info(outlook)


def render_backtest_section():
    st.caption(
        "Looks back through years of history and asks: **every time this trigger fired, "
        "did the stock actually move the right way?** Win% = moved in the strategy's "
        "favour. Signal = ✅ Strong (≥65% win rate) / ⚠️ Mixed / ❌ Weak."
    )
    col_in, col_btn = st.columns([3, 1])
    default_t = st.session_state.get("single_lookup_result_ticker", "")
    bt_ticker = col_in.text_input(
        "Backtest ticker", value=default_t,
        placeholder="e.g. SPY, AAPL, META …",
        label_visibility="collapsed", key="bt_input",
    ).strip().upper()
    run_bt = col_btn.button("🧪 Run Backtest", type="primary",
                            use_container_width=True, key="bt_btn")

    if run_bt and bt_ticker:
        with st.spinner(f"Backtesting **{bt_ticker}** over its full history…"):
            try:
                res = cached_backtest(bt_ticker)
            except ValueError as e:
                st.error(f"{bt_ticker}: {e}")
                return
        if res is None:
            st.error(f"No price history found for **{bt_ticker}** — check the symbol.")
            return
        # Fetch live signals for the summary (2y slice matches screener behaviour)
        import yfinance as _yf
        _hist = _yf.Ticker(bt_ticker).history(period="2y")
        live_sigs = analyze_ticker(bt_ticker, _hist) if _hist is not None and not _hist.empty else None
        st.session_state["bt_result"] = res
        st.session_state["bt_ticker"] = bt_ticker
        st.session_state["bt_live_signals"] = live_sigs

    res = st.session_state.get("bt_result")
    bt_t = st.session_state.get("bt_ticker")
    if not res or not bt_t:
        return

    m = res["meta"]
    st.success(
        f"**{bt_t}** — {m['years']} years of history analysed "
        f"({m['first_date']} → {m['last_date']}, {m['trading_days']:,} trading days)."
    )

    live_sigs = st.session_state.get("bt_live_signals")
    _render_backtest_summary(bt_t, live_sigs, res)

    st.markdown("---")
    st.subheader("Strategy signals — did they work historically?")
    if res["strategies"].empty:
        st.info("No strategy triggers fired in this ticker's history.")
    else:
        simple = _build_simple_bt_df(res["strategies"], "Strategy")
        st.dataframe(_color_simple_bt(simple), use_container_width=True)

    st.subheader("Classic indicator events")
    st.caption(
        'e.g. *"Every time RSI fell below 30, did the stock bounce within 6 weeks?"*'
    )
    if not res["indicators"].empty:
        simple_ind = _build_simple_bt_df(res["indicators"], "Indicator Event")
        st.dataframe(_color_simple_bt(simple_ind), use_container_width=True)

    st.caption(
        "⚠️ **Price-signal backtest** — validates whether entry signals have historically "
        "led to the expected move. Option P&L is not simulated (free data has no historical chains). "
        "Past results do not guarantee future performance."
    )


# ── Welcome screen ────────────────────────────────────────────────────────────

def render_welcome():
    st.info(
        "👆 Click **Run Screen** to start scanning your watchlist "
        "(default: the 78-ticker Goldilocks list — edit it in the sidebar under 📋 Watchlist).\n\n"
        "Expect **2–4 minutes** for the default list due to rate-limiting on free data sources."
    )
    st.subheader("How It Works")
    c1, c2, c3 = st.columns(3)
    c1.info(
        "**Phase 1 — Technicals**\n\n"
        "Fetches 2 years of daily price history for every watchlist ticker. "
        "Computes EMA-50/200 (trend), RSI(9) (momentum), "
        "Bollinger Bands (over/under-extension), and IV Rank (volatility level). "
        "If no strategy trigger fires, **no options chain is fetched** — "
        "saving time and avoiding rate limits."
    )
    c2.info(
        "**Phase 2 — Options Chains**\n\n"
        "Fetches live options data only for triggered tickers. "
        "Applies the **monthly cycle filter** (standard 3rd-Friday expirations only — "
        "no weeklies) and then runs all four liquidity gates before outputting a setup."
    )
    c3.info(
        "**17 Strategies**\n\n"
        "**Credit** (collect premium): Call/Put Credit Spreads, Iron Condor, "
        "Iron Butterfly, Cash-Secured Put, Covered Call, Short Strangle\n\n"
        "**Debit** (pay premium): Bull Call & Bear Put Debit Spreads, "
        "Call/Put Calendars, Long Call, Long Put, Long Straddle, Long Strangle\n\n"
        "Use the sidebar filters to narrow the results."
    )

    st.divider()
    st.subheader("The 4 Liquidity Gates (applied to every setup)")
    g1, g2, g3, g4 = st.columns(4)
    g1.success(
        "**Gate 1 — Price Floor**\n\n"
        "Underlying must close above **$30**. "
        "Sub-$30 names have coarse strike spacing and tiny dollar premiums, "
        "making credit spreads structurally unattractive. "
        "There is no upper cap — mega-caps like META and LLY have some of "
        "the deepest options markets in existence."
    )
    g2.success(
        "**Gate 2 — Bid/Ask Spread**\n\n"
        "The option's bid/ask spread must be **≤ 10% of its mid-price** "
        "(equities) or **≤ $0.05 flat** (index ETFs). "
        "Wide bid/ask = poor liquidity = you lose money on entry and exit slippage."
    )
    g3.success(
        "**Gate 3 — Open Interest**\n\n"
        "The short-strike contract must have **≥ 1,000 open contracts**. "
        "Higher OI ensures a liquid market where you can enter, "
        "adjust, or close the position without moving the price."
    )
    g4.success(
        "**Gate 4 — Yield-to-Risk**\n\n"
        "Net credit collected must be **≥ 20% of the spread width**. "
        "Example: on a $5-wide spread you must collect at least $1.00. "
        "If the setup pays less, the risk/reward is mathematically unattractive "
        "relative to margin requirements."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.title("📊 OTIS — Options Trading Intelligence System")
    st.caption(
        "End-of-Day Screening Engine · Free Data via yfinance · "
        "17 Options Strategies — Credit · Debit · Income · Volatility"
    )

    filters = render_sidebar()

    # ── Restore from disk cache on first load ─────────────────────────────────
    if "results_df" not in st.session_state:
        cached = load_cache()
        if cached:
            st.session_state.update(cached)
        else:
            st.session_state.update({
                "results_df": pd.DataFrame(),
                "signals_data": {},
                "triggered_tickers": [],
                "total_tickers": 0,
                "from_cache": False,
            })

    col_btn, col_notice, col_ts = st.columns([1, 2, 2])
    n_watch = len(filters["tickers"])
    run_clicked = col_btn.button(
        "🚀 Run Screen", type="primary", use_container_width=True,
        help=f"Fetch fresh market data and rerun the full screening pipeline "
             f"for the {n_watch} watchlist ticker{'s' if n_watch != 1 else ''} "
             f"(roughly {max(1, round(n_watch * 2 / 60))}–{max(2, round(n_watch * 3.5 / 60))} minutes).",
    )

    # Staleness check: strikes are anchored to closing prices at run time, so
    # a snapshot from a previous day can show strikes the market has run past.
    last_run_str = st.session_state.get("last_run", "")
    stale_days = None
    if last_run_str:
        try:
            run_dt = datetime.strptime(last_run_str, "%Y-%m-%d %H:%M:%S")
            stale_days = (datetime.now().date() - run_dt.date()).days
        except ValueError:
            pass

    if st.session_state.get("total_tickers", 0) > 0:
        if stale_days is not None and stale_days >= 1:
            col_notice.warning(
                f"**Data is {stale_days} trading day{'s' if stale_days != 1 else ''}+ old** "
                f"(run {last_run_str}). All prices, strikes, and premiums reflect that "
                "snapshot — the market may have moved past them. "
                "Click *Run Screen* for current strikes.",
                icon="⏳",
            )
        elif st.session_state.get("from_cache"):
            col_notice.info(
                "📁 **Showing saved results** — sidebar filters apply instantly. "
                "Click *Run Screen* to fetch fresh market data.",
                icon="📁",
            )

    if last_run_str:
        col_ts.caption(f"Data as of: {last_run_str}")

    if run_clicked:
        st.divider()
        st.subheader("Running Pipeline…")
        cp1, cp2 = st.columns(2)
        cp1.caption("Phase 1: Price History & Technicals (all tickers)")
        cp2.caption("Phase 2: Options Chains (triggered tickers only)")
        ph1_bar = cp1.progress(0)
        ph2_bar = cp2.progress(0)
        status_text = st.empty()

        save_watchlist(filters["tickers"])  # persist custom list across restarts
        try:
            phase1_data, signals_data, triggered_tickers, results_df = run_full_pipeline(
                tickers=tuple(filters["tickers"]),
                r=filters["risk_free_rate"],
                phase1_bar=ph1_bar,
                phase2_bar=ph2_bar,
                status_text=status_text,
            )
            st.session_state.update({
                "results_df": results_df,
                "signals_data": signals_data,
                "triggered_tickers": triggered_tickers,
                "total_tickers": len(phase1_data),
                "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "from_cache": False,
            })
            save_cache(st.session_state)
            n = len(results_df)
            st.success(
                f"Screen complete! {n} qualifying setup{'s' if n != 1 else ''} found. "
                f"Results saved — they'll reload automatically if you navigate away."
            )
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            logger.exception("Pipeline failed")

    # ── News & Events (always available) ──────────────────────────────────────
    st.divider()
    render_news_section()

    # ── Single Ticker Lookup (always available) ───────────────────────────────
    st.divider()
    has_lookup = bool(st.session_state.get("single_lookup_result_ticker"))
    with st.expander(
        "🔍 Single Ticker Lookup",
        expanded=has_lookup,
    ):
        st.caption(
            "Analyze any symbol on demand — runs the full pipeline "
            "(price history → technicals → live options chain → all 17 strategy evaluations). "
            "Results persist here while you adjust sidebar filters. "
            "Not limited to the Goldilocks watchlist."
        )
        render_single_ticker_section(filters)

    # ── Zebra screener (always available) ────────────────────────────────────
    st.divider()
    with st.expander(
        "🦓 Zebra — Stock Replacement Screener",
        expanded=bool(st.session_state.get("zebra_single") or st.session_state.get("zebra_scan")),
    ):
        render_zebra_section(st.session_state.get("signals_data", {}))

    # ── Backtest (always available) ───────────────────────────────────────────
    st.divider()
    with st.expander(
        "🧪 Backtest — how have these signals performed historically?",
        expanded=bool(st.session_state.get("bt_result")),
    ):
        render_backtest_section()

    if st.session_state.get("total_tickers", 0) > 0:
        st.divider()
        render_metrics_row(
            st.session_state["results_df"],
            st.session_state["total_tickers"],
            len(st.session_state["triggered_tickers"]),
            filters,
        )
        st.divider()
        if st.session_state["results_df"].empty:
            st.info(
                "No qualifying setups found with the current settings. "
                "Try relaxing the filters or re-running the screen."
            )
        else:
            render_results_table(
                st.session_state["results_df"],
                filters,
                st.session_state["signals_data"],
            )
    else:
        render_welcome()


if __name__ == "__main__":
    main()
