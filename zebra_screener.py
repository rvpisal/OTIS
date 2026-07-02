"""
OTIS Zebra Screener — Stock-Replacement Strategy (2:1 Call Ratio)

A Zebra (Zero-Extrinsic Back Ratio) is:
  Buy 2× deep ITM call (~80–85 delta)  +  Sell 1× ATM call (~50 delta)
  Same expiration. Net extrinsic ≈ $0.

Why near-zero extrinsic matters: the structure then behaves like owning
100 shares — the 2 long deltas (~1.65 total) minus the short delta (~0.50)
give ~1.15 net delta, with defined max loss (the debit paid) unlike stock.
Capital required is typically 30–70% less than outright stock ownership.

Ideal environment: bullish conviction, low-to-moderate IV (structure is
net long vega — IV crush after entry works against you, unlike credit spreads).
Preferred DTE: 60–120 days (longer than standard credit spreads).
"""

import logging
from datetime import date, datetime

import numpy as np
import pandas as pd
import yfinance as yf

from data_fetcher import is_standard_monthly
from strategy_matrix import get_delta

logger = logging.getLogger(__name__)

# ── Defaults (all overridable via UI filters) ─────────────────────────────────
ZEBRA_DTE_MIN = 60
ZEBRA_DTE_MAX = 120

# Long leg: deep ITM, target delta 0.80–0.85
ZEBRA_LONG_DELTA_LOW    = 0.75
ZEBRA_LONG_DELTA_HIGH   = 0.92
ZEBRA_LONG_DELTA_TARGET = 0.825

# Short leg: ATM, target delta 0.50
ZEBRA_SHORT_DELTA_LOW    = 0.45
ZEBRA_SHORT_DELTA_HIGH   = 0.55
ZEBRA_SHORT_DELTA_TARGET = 0.50

ZEBRA_MIN_OI          = 300    # contracts per leg
ZEBRA_MAX_SPREAD_ABS  = 0.15   # absolute $ bid-ask cap (deep ITM legs are expensive,
                                # so % spread would be too permissive)
ZEBRA_EXTRINSIC_TOL   = 0.10   # max |net extrinsic| per share ($)
ZEBRA_IV_WARN         = 70     # IV rank above this → warn (positive vega at risk)

R = 0.05  # risk-free rate (consistent with strategy_matrix.py)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mid(row: pd.Series) -> float:
    return (float(row["bid"]) + float(row["ask"])) / 2.0


def _extrinsic_call(mid: float, strike: float, spot: float) -> float:
    """Extrinsic value = mid price minus intrinsic (max 0 floor)."""
    return max(0.0, mid - max(0.0, spot - strike))


def _strike_granularity(strikes: list[float]) -> float:
    """Minimum strike increment among available strikes, rounded to $0.01."""
    s = sorted(set(round(x, 2) for x in strikes))
    if len(s) < 2:
        return 10.0
    return min(round(b - a, 2) for a, b in zip(s, s[1:]))


def _passes_spread_gate(row: pd.Series, max_spread: float) -> bool:
    """Absolute dollar bid-ask spread gate (tighter than relative % for deep ITM)."""
    bid = float(row.get("bid", np.nan))
    ask = float(row.get("ask", np.nan))
    if np.isnan(bid) or np.isnan(ask) or ask <= 0:
        return False
    return (ask - bid) <= max_spread


def _passes_oi_gate(row: pd.Series, min_oi: int) -> bool:
    oi = row.get("openInterest", 0)
    return not pd.isna(oi) and int(oi) >= min_oi


def _get_ex_div_date(ticker: str) -> date | None:
    """Next ex-dividend date from yfinance, or None if unavailable/non-dividend."""
    try:
        ts = yf.Ticker(ticker).info.get("exDividendDate")
        if ts and not np.isnan(float(ts)):
            return date.fromtimestamp(int(ts))
    except Exception:
        pass
    return None


# ── Chain fetch ───────────────────────────────────────────────────────────────

def fetch_zebra_chain(
    ticker: str,
    dte_min: int = ZEBRA_DTE_MIN,
    dte_max: int = ZEBRA_DTE_MAX,
) -> dict | None:
    """
    Fetch call-side options chain in the Zebra DTE window (default 60–120 days).
    Restricts to standard monthly expirations for liquidity.
    Returns {"calls": DataFrame} or None.
    """
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            logger.warning(f"[ZEBRA] {ticker}: no expirations")
            return None

        today = date.today()
        all_calls: list[pd.DataFrame] = []

        for exp_str in expirations:
            if not is_standard_monthly(exp_str):
                continue
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if not (dte_min <= dte <= dte_max):
                continue

            calls_df = t.option_chain(exp_str).calls.copy()
            calls_df["expiration"] = exp_str
            calls_df["dte"] = dte
            calls_df["iv_pct"] = calls_df["impliedVolatility"] * 100.0
            all_calls.append(calls_df)

        if not all_calls:
            logger.warning(
                f"[ZEBRA] {ticker}: no standard monthly in {dte_min}–{dte_max} DTE"
            )
            return None

        return {"calls": pd.concat(all_calls, ignore_index=True)}

    except Exception as e:
        logger.error(f"[ZEBRA] {ticker}: fetch_zebra_chain failed — {e}")
        return None


# ── Structure evaluation ──────────────────────────────────────────────────────

def evaluate_zebra(
    ticker: str,
    chain: dict,
    current_price: float,
    signals: dict | None = None,
    earnings_date: date | None = None,
    *,
    min_oi: int = ZEBRA_MIN_OI,
    max_spread: float = ZEBRA_MAX_SPREAD_ABS,
    extrinsic_tol: float = ZEBRA_EXTRINSIC_TOL,
    earnings_warn: bool = True,
    div_warn: bool = True,
) -> list[dict]:
    """
    Scan every qualifying expiration in the chain and return viable Zebra
    structures sorted by |net_extrinsic| (closest to zero first).

    Each result dict contains all fields needed to render the trade card:
    strikes, debit, extrinsic balance, delta, capital metrics, OI/spread,
    strike granularity, and any warning tags.
    """
    calls = chain.get("calls", pd.DataFrame())
    if calls.empty:
        return []

    iv_rank = float((signals or {}).get("iv_rank", 50) or 50)
    ex_div = _get_ex_div_date(ticker) if div_warn else None

    results: list[dict] = []

    for exp, group in calls.groupby("expiration"):
        dte = int(group["dte"].iloc[0])
        gran = _strike_granularity(group["strike"].tolist())

        # ── Find best long leg (~80–85 delta, deep ITM) ───────────────────────
        long_cands: list[tuple[float, float, pd.Series]] = []
        for _, row in group.iterrows():
            K = float(row["strike"])
            if K <= 0:
                continue
            if not _passes_oi_gate(row, min_oi):
                continue
            if not _passes_spread_gate(row, max_spread):
                continue
            raw_iv = row.get("iv_pct")
            iv_pct = float(raw_iv) if pd.notna(raw_iv) and float(raw_iv) > 0 else None
            d = get_delta("c", current_price, K, iv_pct, dte, R)
            if ZEBRA_LONG_DELTA_LOW <= d <= ZEBRA_LONG_DELTA_HIGH:
                long_cands.append((abs(d - ZEBRA_LONG_DELTA_TARGET), d, row.copy()))

        # ── Find best short leg (~50 delta, ATM) ──────────────────────────────
        short_cands: list[tuple[float, float, pd.Series]] = []
        for _, row in group.iterrows():
            K = float(row["strike"])
            if K <= 0:
                continue
            if not _passes_oi_gate(row, min_oi):
                continue
            if not _passes_spread_gate(row, max_spread):
                continue
            raw_iv = row.get("iv_pct")
            iv_pct = float(raw_iv) if pd.notna(raw_iv) and float(raw_iv) > 0 else None
            d = get_delta("c", current_price, K, iv_pct, dte, R)
            if ZEBRA_SHORT_DELTA_LOW <= d <= ZEBRA_SHORT_DELTA_HIGH:
                short_cands.append((abs(d - ZEBRA_SHORT_DELTA_TARGET), d, row.copy()))

        if not long_cands or not short_cands:
            continue

        long_cands.sort(key=lambda x: x[0])
        short_cands.sort(key=lambda x: x[0])
        _, long_delta, long_row = long_cands[0]
        _, short_delta, short_row = short_cands[0]

        long_strike  = float(long_row["strike"])
        short_strike = float(short_row["strike"])

        # Long must sit below ATM short (deep ITM < ATM)
        if long_strike >= short_strike:
            continue

        long_mid  = _mid(long_row)
        short_mid = _mid(short_row)

        long_ext  = _extrinsic_call(long_mid, long_strike, current_price)
        short_ext = _extrinsic_call(short_mid, short_strike, current_price)
        net_ext   = round(2 * long_ext - short_ext, 3)  # target ≈ $0

        if abs(net_ext) > extrinsic_tol:
            continue  # extrinsic balance too far from zero

        net_debit = round(2 * long_mid - short_mid, 2)
        if net_debit <= 0:
            continue  # net credit — degenerate structure

        structure_delta    = round(2 * long_delta - short_delta, 2)
        capital_req        = round(net_debit * 100, 2)      # per 1 structure = 100 shs
        share_cost         = round(current_price * 100, 2)
        pct_saved          = round((share_cost - capital_req) / share_cost * 100, 1) if share_cost > 0 else 0.0
        breakeven          = round(long_strike + net_debit, 2)

        # ── Build warning tags ────────────────────────────────────────────────
        warnings: list[str] = []

        if iv_rank > ZEBRA_IV_WARN:
            warnings.append(
                f"High IV Rank ({iv_rank:.0f}) — positive vega at risk if IV contracts post-entry"
            )

        exp_date = datetime.strptime(str(exp), "%Y-%m-%d").date()

        if earnings_date and earnings_warn:
            days_earn = (earnings_date - date.today()).days
            if earnings_date <= exp_date:
                warnings.append(
                    f"Earnings {earnings_date} ({days_earn}d) inside trade window — "
                    "disable warning if the vol event IS your thesis"
                )
            elif days_earn <= 5:
                warnings.append(
                    f"Earnings in {days_earn}d — within 5-day proximity flag"
                )

        if ex_div and ex_div <= exp_date:
            warnings.append(
                f"Ex-dividend {ex_div} before expiry — early assignment risk on deep ITM long calls"
            )

        if gran > 2.5:
            warnings.append(
                f"${gran} strike increments in this zone — coarse grid may prevent hitting near-zero extrinsic"
            )

        results.append({
            "ticker":             ticker,
            "current_price":      round(current_price, 2),
            "expiration":         str(exp),
            "dte":                dte,
            "long_strike":        long_strike,
            "short_strike":       short_strike,
            "long_delta":         round(long_delta, 2),
            "short_delta":        round(short_delta, 2),
            "structure_delta":    structure_delta,
            "long_mid":           round(long_mid, 2),
            "short_mid":          round(short_mid, 2),
            "long_bid":           round(float(long_row.get("bid", 0)), 2),
            "long_ask":           round(float(long_row.get("ask", 0)), 2),
            "short_bid":          round(float(short_row.get("bid", 0)), 2),
            "short_ask":          round(float(short_row.get("ask", 0)), 2),
            "long_ext":           round(long_ext, 2),
            "short_ext":          round(short_ext, 2),
            "net_extrinsic":      net_ext,
            "net_debit":          net_debit,
            "breakeven":          breakeven,
            "max_loss":           capital_req,
            "capital_required":   capital_req,
            "share_cost":         share_cost,
            "capital_efficiency": pct_saved,
            "long_oi":            int(long_row.get("openInterest", 0)),
            "short_oi":           int(short_row.get("openInterest", 0)),
            "long_spread":        round(float(long_row.get("ask", 0)) - float(long_row.get("bid", 0)), 2),
            "short_spread":       round(float(short_row.get("ask", 0)) - float(short_row.get("bid", 0)), 2),
            "strike_granularity": gran,
            "iv_rank":            round(iv_rank, 1),
            "warnings":           warnings,
        })

    return sorted(results, key=lambda r: abs(r["net_extrinsic"]))


if __name__ == "__main__":
    import time
    print("Fetching AAPL Zebra chain (60–120 DTE)…")
    chain = fetch_zebra_chain("AAPL")
    if chain is None:
        print("No chain found.")
    else:
        price = float(yf.Ticker("AAPL").history(period="2d")["Close"].iloc[-1])
        print(f"AAPL @ ${price:.2f}")
        results = evaluate_zebra("AAPL", chain, price)
        if not results:
            print("No viable Zebra structures found (check OI / spread / extrinsic gates).")
        for r in results[:3]:
            print(
                f"  Buy 2× ${r['long_strike']}C + Sell 1× ${r['short_strike']}C  "
                f"exp {r['expiration']} ({r['dte']}d)  "
                f"debit=${r['net_debit']}  net_ext=${r['net_extrinsic']}  "
                f"Δ={r['structure_delta']}  cap_eff={r['capital_efficiency']}%  "
                f"warnings={r['warnings']}"
            )
