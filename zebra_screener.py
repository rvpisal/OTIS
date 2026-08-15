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
) -> list[dict]:
    """
    Return ALL structurally valid Zebra setups across qualifying expirations,
    regardless of liquidity or extrinsic filters.

    Only hard structural gates are applied here (they describe invalid trades,
    not just ones that don't meet the current filter settings):
      - Must find a strike in the long-delta range (0.75–0.92)
      - Must find a strike in the short-delta range (0.45–0.55)
      - Long strike must be below the short (deep ITM < ATM)
      - Net debit must be positive (2×long − short > 0)

    Soft gates — OI, bid-ask spread, extrinsic tolerance, earnings proximity —
    are NOT applied here. They are stamped as `filter_status` by
    `apply_zebra_filters()` so the UI can show every candidate and explain
    exactly what each one fails.
    """
    calls = chain.get("calls", pd.DataFrame())
    if calls.empty:
        return []

    iv_rank = float((signals or {}).get("iv_rank", 50) or 50)
    ex_div  = _get_ex_div_date(ticker)

    results: list[dict] = []

    for exp, group in calls.groupby("expiration"):
        dte  = int(group["dte"].iloc[0])
        gran = _strike_granularity(group["strike"].tolist())

        # ── Candidate search: delta only, NO liquidity gates ──────────────────
        long_cands: list[tuple[float, float, pd.Series]] = []
        short_cands: list[tuple[float, float, pd.Series]] = []

        for _, row in group.iterrows():
            K = float(row["strike"])
            if K <= 0:
                continue
            raw_iv = row.get("iv_pct")
            iv_pct = float(raw_iv) if pd.notna(raw_iv) and float(raw_iv) > 0 else None
            d = get_delta("c", current_price, K, iv_pct, dte, R)
            if ZEBRA_LONG_DELTA_LOW <= d <= ZEBRA_LONG_DELTA_HIGH:
                long_cands.append((abs(d - ZEBRA_LONG_DELTA_TARGET), d, row.copy()))
            if ZEBRA_SHORT_DELTA_LOW <= d <= ZEBRA_SHORT_DELTA_HIGH:
                short_cands.append((abs(d - ZEBRA_SHORT_DELTA_TARGET), d, row.copy()))

        if not long_cands or not short_cands:
            continue  # no strikes in the required delta zones — hard skip

        long_cands.sort(key=lambda x: x[0])
        short_cands.sort(key=lambda x: x[0])
        _, long_delta, long_row  = long_cands[0]
        _, short_delta, short_row = short_cands[0]

        long_strike  = float(long_row["strike"])
        short_strike = float(short_row["strike"])

        if long_strike >= short_strike:
            continue  # degenerate — hard skip

        long_mid  = _mid(long_row)
        short_mid = _mid(short_row)
        net_debit = round(2 * long_mid - short_mid, 2)

        if net_debit <= 0:
            continue  # net credit structure — hard skip

        long_ext  = _extrinsic_call(long_mid, long_strike, current_price)
        short_ext = _extrinsic_call(short_mid, short_strike, current_price)
        net_ext   = round(2 * long_ext - short_ext, 3)

        structure_delta = round(2 * long_delta - short_delta, 2)
        capital_req     = round(net_debit * 100, 2)
        share_cost      = round(current_price * 100, 2)
        pct_saved       = round((share_cost - capital_req) / share_cost * 100, 1) if share_cost > 0 else 0.0
        breakeven       = round(long_strike + net_debit, 2)

        long_oi     = int(long_row.get("openInterest", 0))
        short_oi    = int(short_row.get("openInterest", 0))
        long_spread  = round(float(long_row.get("ask", 0)) - float(long_row.get("bid", 0)), 2)
        short_spread = round(float(short_row.get("ask", 0)) - float(short_row.get("bid", 0)), 2)

        # ── Risk warnings (IV, earnings, ex-div) — always stored ─────────────
        warnings: list[str] = []
        if iv_rank > ZEBRA_IV_WARN:
            warnings.append(
                f"High IV Rank ({iv_rank:.0f}) — positive vega at risk if IV contracts post-entry"
            )
        exp_date = datetime.strptime(str(exp), "%Y-%m-%d").date()
        if earnings_date:
            days_earn = (earnings_date - date.today()).days
            if earnings_date <= exp_date:
                warnings.append(
                    f"Earnings {earnings_date} ({days_earn}d) inside trade window — "
                    "disable if vol event is your thesis"
                )
            elif days_earn <= 5:
                warnings.append(f"Earnings in {days_earn}d — within 5-day proximity flag")
        if ex_div and ex_div <= exp_date:
            warnings.append(
                f"Ex-dividend {ex_div} before expiry — early assignment risk on deep ITM longs"
            )
        if gran > 2.5:
            warnings.append(
                f"${gran} strike increments — coarse grid may prevent near-zero extrinsic balance"
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
            "long_oi":            long_oi,
            "short_oi":           short_oi,
            "long_spread":        long_spread,
            "short_spread":       short_spread,
            "strike_granularity": gran,
            "iv_rank":            round(iv_rank, 1),
            "warnings":           warnings,
            "filter_status":      "",   # filled by apply_zebra_filters()
            # stored for apply_zebra_filters earnings toggle
            "_earnings_date":     earnings_date,
            "_ex_div":            ex_div,
        })

    return sorted(results, key=lambda r: abs(r["net_extrinsic"]))


def apply_zebra_filters(
    results: list[dict],
    *,
    min_oi: int = ZEBRA_MIN_OI,
    max_spread: float = ZEBRA_MAX_SPREAD_ABS,
    extrinsic_tol: float = ZEBRA_EXTRINSIC_TOL,
    earnings_warn: bool = True,
    div_warn: bool = True,
) -> list[dict]:
    """
    Stamp filter_status on every result, mirroring how add_filter_flags() works
    in the main screener.  Does NOT remove rows — every structure remains visible
    so the user can see exactly what it fails.

    "✅ Passes all filters"  — all soft gates pass
    "⚠️ <reason> · <reason>" — one or more gates miss, each explained precisely
    """
    out = []
    for r in results:
        issues: list[str] = []

        if abs(r["net_extrinsic"]) > extrinsic_tol:
            issues.append(
                f"net extrinsic ${r['net_extrinsic']:+.2f}/shr "
                f"(limit ±${extrinsic_tol:.2f})"
            )
        if r["long_oi"] < min_oi:
            issues.append(f"long OI {r['long_oi']:,} < {min_oi:,} required")
        if r["short_oi"] < min_oi:
            issues.append(f"short OI {r['short_oi']:,} < {min_oi:,} required")
        if r["long_spread"] > max_spread:
            issues.append(
                f"long spread ${r['long_spread']:.2f} > ${max_spread:.2f} limit"
            )
        if r["short_spread"] > max_spread:
            issues.append(
                f"short spread ${r['short_spread']:.2f} > ${max_spread:.2f} limit"
            )

        # Earnings proximity is a toggleable filter (user may want the vol event)
        if earnings_warn:
            ed = r.get("_earnings_date")
            if ed:
                try:
                    exp_d = datetime.strptime(r["expiration"], "%Y-%m-%d").date()
                    days_earn = (ed - date.today()).days
                    if ed <= exp_d:
                        issues.append(
                            f"earnings {ed} ({days_earn}d) inside window"
                        )
                except (ValueError, TypeError):
                    pass

        # Dividend risk flag (toggleable)
        if div_warn:
            ex_div = r.get("_ex_div")
            if ex_div:
                try:
                    exp_d = datetime.strptime(r["expiration"], "%Y-%m-%d").date()
                    if ex_div <= exp_d:
                        issues.append(f"ex-div {ex_div} before expiry")
                except (ValueError, TypeError):
                    pass

        r = dict(r)  # shallow copy — don't mutate the cached list
        r["filter_status"] = "✅ Passes all filters" if not issues else "⚠️ " + " · ".join(issues)
        out.append(r)

    # ✅ rows first, then ⚠️ — within each group keep |net_extrinsic| order
    return sorted(out, key=lambda x: (0 if x["filter_status"].startswith("✅") else 1, abs(x["net_extrinsic"])))


# ── Entry confirmation scoring ────────────────────────────────────────────────

def score_entry_confirmation(
    signals: dict,
    history_df: pd.DataFrame,
    bias: str = "bullish",
) -> dict:
    """
    Score entry quality across 5 criteria and return a summary dict.

    Criteria (each worth 1 point; pullback-hold can be 0.5 if pending):
      1. Structural level break — price ≥ 20d rolling high (bullish) / ≤ 20d low (bearish)
      2. Volume confirmation   — latest session volume > 1.2× 20d average
      3. MA alignment          — 20d SMA slope up AND 50d EMA slope up (bullish)
      4. Pullback hold         — price pulled back to SMA20 and recovered (pending = 0.5)
      5. RSI divergence        — RSI9 > 50 for bullish, or higher-low RSI divergence

    Returns:
        {
          "status":       "confirmed" | "thesis_only" | "conflicting",
          "status_emoji": "🟢" | "🟡" | "🔴",
          "score":        int,        # 0–5 (rounded from 0.5 increments)
          "score_raw":    float,
          "criteria": {
              "structural_break": {"pass": bool, "label": str},
              "volume_confirm":   {"pass": bool, "label": str},
              "ma_alignment":     {"pass": bool, "label": str},
              "pullback_hold":    {"pass": bool | None, "label": str, "pending": bool},
              "rsi_divergence":   {"pass": bool, "label": str},
          }
        }
    """
    try:
        close  = history_df["Close"].dropna()
        high   = history_df["High"].dropna()
        low    = history_df["Low"].dropna()
        volume = history_df["Volume"].dropna()

        if len(close) < 55:
            raise ValueError("Insufficient price history for entry scoring")

        price           = float(close.iloc[-1])
        sma20           = close.rolling(20).mean()
        sma20_current   = float(sma20.iloc[-1])
        sma20_5d_ago    = float(sma20.iloc[-6]) if len(sma20) > 6 else sma20_current

        # ── 1. Structural level break ─────────────────────────────────────────
        high20_5d_ago = float(high.rolling(20).max().iloc[-6])
        low20_5d_ago  = float(low.rolling(20).min().iloc[-6])
        if bias == "bullish":
            struct_pass = price >= high20_5d_ago * 0.995   # 0.5% tolerance
            struct_label = (
                f"Price ${price:.2f} ≥ 20d high ${high20_5d_ago:.2f} — breakout confirmed ✓"
                if struct_pass
                else f"Price ${price:.2f} < 20d high ${high20_5d_ago:.2f} — no breakout yet"
            )
        else:
            struct_pass = price <= low20_5d_ago * 1.005
            struct_label = (
                f"Price ${price:.2f} ≤ 20d low ${low20_5d_ago:.2f} — breakdown confirmed ✓"
                if struct_pass
                else f"Price ${price:.2f} > 20d low ${low20_5d_ago:.2f} — no breakdown yet"
            )

        # ── 2. Volume confirmation ─────────────────────────────────────────────
        avg_vol20 = float(volume.rolling(20).mean().iloc[-1])
        last_vol  = float(volume.iloc[-1])
        vol_ratio = last_vol / avg_vol20 if avg_vol20 > 0 else 0.0
        vol_pass  = vol_ratio >= 1.2
        vol_label = (
            f"Volume {vol_ratio:.1f}× 20d avg — confirmation present ✓"
            if vol_pass
            else f"Volume {vol_ratio:.1f}× 20d avg (< 1.2× needed)"
        )

        # ── 3. MA alignment ───────────────────────────────────────────────────
        sma20_slope_up = sma20_current > sma20_5d_ago
        ema50_series   = close.ewm(span=50, adjust=False).mean()
        ema50_current  = float(ema50_series.iloc[-1])
        ema50_5d_ago   = float(ema50_series.iloc[-6])
        ema50_slope_up = ema50_current > ema50_5d_ago

        if bias == "bullish":
            ma_pass  = sma20_slope_up and ema50_slope_up
        else:
            ma_pass  = (not sma20_slope_up) and (not ema50_slope_up)
        ma_label = (
            f"SMA20 {'▲' if sma20_slope_up else '▼'} · EMA50 {'▲' if ema50_slope_up else '▼'} "
            f"({'aligned ✓' if ma_pass else f'not aligned for {bias}'})"
        )

        # ── 4. Pullback hold ──────────────────────────────────────────────────
        recent_low3  = float(low.iloc[-3:].min())
        recent_high3 = float(high.iloc[-3:].max())
        pb_pending   = False

        if bias == "bullish":
            touched_sma = recent_low3 <= sma20_current * 1.02
            back_above  = price >= sma20_current
            if touched_sma and back_above:
                pb_pass  = True
                pb_label = f"Pulled back to SMA20 ${sma20_current:.2f} and recovered ✓"
            elif back_above:
                pb_pass    = None   # no pullback yet — holding above
                pb_pending = True
                pb_label   = f"Holding above SMA20 ${sma20_current:.2f} — no pullback yet (pending)"
            else:
                pb_pass  = False
                pb_label = f"Price ${price:.2f} below SMA20 ${sma20_current:.2f}"
        else:
            touched_sma = recent_high3 >= sma20_current * 0.98
            back_below  = price <= sma20_current
            if touched_sma and back_below:
                pb_pass  = True
                pb_label = f"Bounced to SMA20 ${sma20_current:.2f} and rejected ✓"
            elif back_below:
                pb_pass    = None
                pb_pending = True
                pb_label   = f"Holding below SMA20 ${sma20_current:.2f} — no bounce yet (pending)"
            else:
                pb_pass  = False
                pb_label = f"Price ${price:.2f} above SMA20 ${sma20_current:.2f}"

        # ── 5. RSI divergence ─────────────────────────────────────────────────
        rsi9        = float(signals.get("rsi9") or 50)
        delta_s     = close.diff()
        gain        = delta_s.clip(lower=0).rolling(9).mean()
        loss        = (-delta_s.clip(upper=0)).rolling(9).mean()
        rs          = gain / loss.replace(0, np.nan)
        rsi_series  = 100 - (100 / (1 + rs))
        rsi_5d_ago  = float(rsi_series.iloc[-6]) if len(rsi_series) > 6 else rsi9
        price_5d_ago = float(close.iloc[-6]) if len(close) > 6 else price

        if bias == "bullish":
            if rsi9 > 50:
                rsi_pass  = True
                rsi_label = f"RSI9 {rsi9:.0f} > 50 — momentum confirms bullish ✓"
            elif price < price_5d_ago and rsi9 > rsi_5d_ago:
                rsi_pass  = True
                rsi_label = (
                    f"RSI9 {rsi9:.0f} vs {rsi_5d_ago:.0f} (rising while price pulled back) "
                    "— bullish divergence ✓"
                )
            else:
                rsi_pass  = False
                rsi_label = f"RSI9 {rsi9:.0f} < 50 and no divergence detected"
        else:
            if rsi9 < 50:
                rsi_pass  = True
                rsi_label = f"RSI9 {rsi9:.0f} < 50 — momentum confirms bearish ✓"
            elif price > price_5d_ago and rsi9 < rsi_5d_ago:
                rsi_pass  = True
                rsi_label = (
                    f"RSI9 {rsi9:.0f} vs {rsi_5d_ago:.0f} (falling while price bounced) "
                    "— bearish divergence ✓"
                )
            else:
                rsi_pass  = False
                rsi_label = f"RSI9 {rsi9:.0f} > 50 and no divergence detected"

        # ── Score & overall status ────────────────────────────────────────────
        score_raw = (
            (1.0 if struct_pass else 0.0)
            + (1.0 if vol_pass else 0.0)
            + (1.0 if ma_pass else 0.0)
            + (0.5 if pb_pass is None else (1.0 if pb_pass else 0.0))
            + (1.0 if rsi_pass else 0.0)
        )
        score = int(round(score_raw))

        if score >= 4:
            status = "confirmed"
            emoji  = "🟢"
        elif score >= 3:
            status = "thesis_only"
            emoji  = "🟡"
        else:
            status = "conflicting"
            emoji  = "🔴"

        return {
            "status":       status,
            "status_emoji": emoji,
            "score":        score,
            "score_raw":    score_raw,
            "criteria": {
                "structural_break": {"pass": struct_pass, "label": struct_label},
                "volume_confirm":   {"pass": vol_pass,    "label": vol_label},
                "ma_alignment":     {"pass": ma_pass,     "label": ma_label},
                "pullback_hold":    {"pass": pb_pass,     "label": pb_label,  "pending": pb_pending},
                "rsi_divergence":   {"pass": rsi_pass,    "label": rsi_label},
            },
        }

    except Exception as e:
        logger.error(f"score_entry_confirmation failed: {e}")
        return {
            "status":       "conflicting",
            "status_emoji": "🔴",
            "score":        0,
            "score_raw":    0.0,
            "criteria":     {},
        }


def get_time_decay_status(
    current_price: float,
    breakeven: float,
    dte: int,
) -> dict:
    """
    Diagnose whether flat price is costing theta on this Zebra position.

    The Zebra is NOT theta-neutral: the short ATM call decays fastest.
    When price is flat near breakeven the short leg's theta erodes value
    without the long legs gaining intrinsic fast enough — unlike actual stock.

    Returns: {"status": "costly" | "fine", "icon": str, "label": str}
    """
    gap_pct = (current_price - breakeven) / breakeven if breakeven > 0 else 0.0

    if dte > 21 and gap_pct < 0.05:          # within 5% of breakeven with time left
        return {
            "status": "costly",
            "icon":   "⚠️",
            "label": (
                f"Flat price is costing you time value — spot ${current_price:.2f} is within 5% "
                f"of breakeven ${breakeven:.2f} with {dte}d remaining. "
                "Structure needs upward movement to behave like stock."
            ),
        }
    direction = "above" if gap_pct >= 0 else "below"
    note = " Watch theta as DTE decreases." if 0 <= gap_pct < 0.10 and dte > 21 else ""
    return {
        "status": "fine",
        "icon":   "✓",
        "label": (
            f"Position behaving close to stock-equivalent — "
            f"spot ${current_price:.2f} is {abs(gap_pct)*100:.1f}% {direction} "
            f"breakeven ${breakeven:.2f}.{note}"
        ),
    }


if __name__ == "__main__":
    import time
    print("Fetching AAPL Zebra chain (60–120 DTE)…")
    chain = fetch_zebra_chain("AAPL")
    if chain is None:
        print("No chain found.")
    else:
        price = float(yf.Ticker("AAPL").history(period="2d")["Close"].iloc[-1])
        print(f"AAPL @ ${price:.2f}")
        results = apply_zebra_filters(evaluate_zebra("AAPL", chain, price))
        if not results:
            print("No structurally valid Zebra setups found.")
        for r in results[:3]:
            print(
                f"  Buy 2× ${r['long_strike']}C + Sell 1× ${r['short_strike']}C  "
                f"exp {r['expiration']} ({r['dte']}d)  "
                f"debit=${r['net_debit']}  net_ext=${r['net_extrinsic']}  "
                f"Δ={r['structure_delta']}  cap_eff={r['capital_efficiency']}%  "
                f"warnings={r['warnings']}"
            )
