"""
Signal backtester for OTIS.

Answers: "when this trigger fired historically, what did the stock do next?"

Design principles:
  - ZERO trigger drift: evaluates the SAME `STRATEGY_TRIGGERS` lambdas the live
    screener uses (indicators.py), against a historical signals DataFrame whose
    columns match the live signal-dict keys.
  - Signal events = trigger ONSET days only (False→True transition), so one
    15-day oversold episode counts once, not 15 times.
  - No look-ahead: the event at day i is scored on returns from i to i+h.

Honesty constraint: free historical option chains do not exist, so this is a
PRICE-signal backtest. It validates the entry signals (trend, RSI, IV regime),
not actual spread P&L.
"""

import logging

import numpy as np
import pandas as pd
import ta

from indicators import STRATEGY_TRIGGERS
from strategy_matrix import STRATEGY_META

logger = logging.getLogger(__name__)

HORIZONS = (5, 10, 21, 45)  # trading days
WARMUP_ROWS = 252           # need a year of data before signals are valid

# Directional expectation for classic indicator events (mean-reversion /
# trend-following reading). "win" = forward return in the expected direction.
INDICATOR_EVENTS = {
    "RSI(9) < 30 (oversold)":        {"expect": "up"},
    "RSI(9) > 70 (overbought)":      {"expect": "down"},
    "Lower Bollinger Band touch":    {"expect": "up"},
    "Upper Bollinger Band touch":    {"expect": "down"},
    "Golden cross (EMA50 > EMA200)": {"expect": "up"},
    "Death cross (EMA50 < EMA200)":  {"expect": "down"},
}


def build_signal_history(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full indicator series over a price history, producing one row
    per trading day with the same keys as the live signal bundle
    (TechnicalAnalyzer.run_all), so STRATEGY_TRIGGERS evaluate identically.
    """
    close = history_df["Close"]

    ema50 = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
    ema200 = ta.trend.EMAIndicator(close=close, window=200).ema_indicator()
    rsi9 = ta.momentum.RSIIndicator(close=close, window=9).rsi()

    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()

    # Macro trend with the same ±0.5% neutral band as compute_macro_trend
    threshold = ema200 * 0.005
    macro_trend = pd.Series(
        np.where(ema50 > ema200 + threshold, "BULLISH",
                 np.where(ema50 < ema200 - threshold, "BEARISH", "NEUTRAL")),
        index=close.index,
    )

    # IV Rank proxy: 21-day HV ranked in its trailing 252-day min/max range —
    # the same formula the live system falls back to when no chain exists.
    log_ret = np.log(close / close.shift(1))
    hv = log_ret.rolling(21).std() * np.sqrt(252)
    hv_min = hv.rolling(252).min()
    hv_max = hv.rolling(252).max()
    rng = (hv_max - hv_min).replace(0, np.nan)
    iv_rank = ((hv - hv_min) / rng * 100).clip(0, 100).fillna(50.0)

    sig = pd.DataFrame({
        "current_price": close,
        "macro_trend": macro_trend,
        "ema50": ema50,
        "ema200": ema200,
        "rsi9": rsi9,
        "iv_rank": iv_rank,
        "high_vol_env": iv_rank > 45.0,
        "bb_upper_touch": close >= bb_upper,
        "bb_lower_touch": close <= bb_lower,
    })
    return sig.iloc[WARMUP_ROWS:].dropna(subset=["rsi9", "ema200", "iv_rank"])


def _forward_returns(close: pd.Series, horizons=HORIZONS) -> pd.DataFrame:
    return pd.DataFrame(
        {h: close.shift(-h) / close - 1.0 for h in horizons}, index=close.index
    )


def _onsets(trigger: pd.Series) -> pd.Series:
    """True only on the day a trigger flips from False to True."""
    return trigger & ~trigger.shift(1, fill_value=False)


def _win_mask(fwd: pd.Series, bias: str) -> pd.Series:
    if bias == "bullish":
        return fwd > 0
    if bias == "bearish":
        return fwd < 0
    if bias == "neutral":
        return fwd.abs() <= 0.03   # range held within ±3%
    return fwd.abs() >= 0.05       # volatility: breakout of ≥5% happened


def _event_stats(event_days: pd.Series, fwd_df: pd.DataFrame,
                 win_fn) -> dict | None:
    """Win rate / avg / worst per horizon for one boolean event series."""
    idx = event_days[event_days].index
    if len(idx) == 0:
        return None
    row: dict = {"occurrences": int(len(idx))}
    for h in fwd_df.columns:
        fwd = fwd_df.loc[idx, h].dropna()  # drop events too near data end
        if fwd.empty:
            row[f"win_{h}d"] = np.nan
            row[f"avg_{h}d"] = np.nan
            continue
        wins = win_fn(fwd)
        row[f"win_{h}d"] = round(float(wins.mean()) * 100, 1)
        row[f"avg_{h}d"] = round(float(fwd.mean()) * 100, 2)
    longest = fwd_df.columns[-1]
    fwd_long = fwd_df.loc[idx, longest].dropna()
    row[f"worst_{longest}d"] = round(float(fwd_long.min()) * 100, 1) if not fwd_long.empty else np.nan
    row[f"best_{longest}d"] = round(float(fwd_long.max()) * 100, 1) if not fwd_long.empty else np.nan
    return row


def backtest_ticker(history_df: pd.DataFrame, horizons=HORIZONS) -> dict:
    """
    Full signal backtest over a price history (use period="max" for depth).

    Returns:
      {
        "meta":       {first_date, last_date, years, trading_days},
        "strategies": DataFrame — one row per strategy trigger,
        "indicators": DataFrame — one row per classic indicator event,
      }
    """
    if history_df is None or len(history_df) < WARMUP_ROWS + max(horizons) + 20:
        raise ValueError(
            f"Need at least ~{WARMUP_ROWS + max(horizons) + 20} trading days "
            f"of history (got {0 if history_df is None else len(history_df)})"
        )

    sig = build_signal_history(history_df)
    close = sig["current_price"]
    fwd_df = _forward_returns(close, horizons)

    # ── Strategy triggers — the exact live lambdas, evaluated row-wise ────────
    strat_rows = {}
    for key, trigger in STRATEGY_TRIGGERS.items():
        fired = sig.apply(trigger, axis=1)
        stats = _event_stats(
            _onsets(fired), fwd_df,
            win_fn=lambda f, b=STRATEGY_META[key]["bias"]: _win_mask(f, b),
        )
        if stats:
            meta = STRATEGY_META[key]
            strat_rows[f"{meta['emoji']} {meta['name']}"] = {"bias": meta["bias"], **stats}

    # ── Classic indicator events ──────────────────────────────────────────────
    golden = (sig["ema50"] > sig["ema200"])
    ind_series = {
        "RSI(9) < 30 (oversold)":        sig["rsi9"] < 30,
        "RSI(9) > 70 (overbought)":      sig["rsi9"] > 70,
        "Lower Bollinger Band touch":    sig["bb_lower_touch"],
        "Upper Bollinger Band touch":    sig["bb_upper_touch"],
        "Golden cross (EMA50 > EMA200)": golden,
        "Death cross (EMA50 < EMA200)":  ~golden,
    }
    ind_rows = {}
    for name, series in ind_series.items():
        expect = INDICATOR_EVENTS[name]["expect"]
        stats = _event_stats(
            _onsets(series), fwd_df,
            win_fn=(lambda f: f > 0) if expect == "up" else (lambda f: f < 0),
        )
        if stats:
            ind_rows[name] = {"expectation": expect, **stats}

    first, last = sig.index[0], sig.index[-1]
    return {
        "meta": {
            "first_date": str(first.date()) if hasattr(first, "date") else str(first),
            "last_date": str(last.date()) if hasattr(last, "date") else str(last),
            "trading_days": int(len(sig)),
            "years": round(len(sig) / 252, 1),
        },
        "strategies": pd.DataFrame.from_dict(strat_rows, orient="index"),
        "indicators": pd.DataFrame.from_dict(ind_rows, orient="index"),
    }


if __name__ == "__main__":
    import yfinance as yf

    print("Fetching SPY max history...")
    df = yf.Ticker("SPY").history(period="max")
    print(f"  {len(df)} trading days ({df.index[0].date()} → {df.index[-1].date()})")

    result = backtest_ticker(df)
    m = result["meta"]
    print(f"\nBacktest window: {m['first_date']} → {m['last_date']} "
          f"({m['years']} years, {m['trading_days']} days)\n")

    pd.set_option("display.width", 200)
    print("── Strategy triggers ──")
    print(result["strategies"].to_string())
    print("\n── Indicator events ──")
    print(result["indicators"].to_string())

    # Sanity assertions
    for table in (result["strategies"], result["indicators"]):
        win_cols = [c for c in table.columns if c.startswith("win_")]
        vals = table[win_cols].to_numpy(dtype=float)
        assert np.nanmin(vals) >= 0 and np.nanmax(vals) <= 100, "win rate out of range"
    assert (result["indicators"].loc["RSI(9) < 30 (oversold)", "occurrences"] > 10)
    print("\nSanity checks passed ✓")
