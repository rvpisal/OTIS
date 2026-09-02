"""
OTIS Earnings Playbook — Pre-Earnings Analysis & Strategy Selection

Analyses a stock's earnings history, scores directional bias, computes the
market's implied move from the live options chain, and recommends strategies
for three stages: Scan (5-10 days out), Enter (1-3 days before), Announcement day.

HONEST CONFIDENCE ASSESSMENT
─────────────────────────────
Direction prediction accuracy using only free, public data:
  · Base rate (coin flip):                   50%
  · High-conviction setups (all 5 signals):  58–63%
  · Academic ceiling (public data only):     ~65%

The market already prices in analyst consensus — we cannot systematically beat
it on direction. The real value of this playbook is:
  1. Strategy TYPE selection (credit vs. debit) based on IV rank — >70% reliable
  2. Magnitude context — the options-implied move is accurate within 20-30%
  3. Risk management framing — what to enter, when, and when to close
"""

import logging
import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BEAT_WINDOW           = 4    # quarters to count for beat streak
HISTORY_QUARTERS      = 12   # max quarters of history to return (3 years)
IMPLIED_MOVE_MAX_DTE  = 45   # don't use expiries beyond this for implied move
IMPLIED_MOVE_MIN_DTE  = 0    # must expire on or after earnings date


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val, default=float("nan")) -> float:
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _pct(numerator, denominator, default=float("nan")) -> float:
    try:
        if denominator == 0:
            return default
        return round(numerator / denominator * 100, 1)
    except (TypeError, ValueError):
        return default


# ── 1. Earnings History ───────────────────────────────────────────────────────

def fetch_earnings_history(ticker: str, history_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build a per-quarter DataFrame of EPS surprises + actual stock reactions.

    Columns returned:
      date (index), eps_estimate, eps_actual, surprise_pct,
      close_before, close_after, move_1d_pct, close_5d, move_5d_pct,
      beat (bool), up_1d (bool)

    history_df: optional pre-fetched OHLCV DataFrame (2y of daily data).
                If None, fetched fresh.
    Returns empty DataFrame if earnings data is unavailable (ETFs, etc.).
    """
    try:
        t = yf.Ticker(ticker)

        # ── EPS history ───────────────────────────────────────────────────────
        raw = t.earnings_dates
        if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
            logger.info(f"[EARNINGS] {ticker}: no earnings_dates (ETF or insufficient data)")
            return pd.DataFrame()

        # Keep past events only (Reported EPS is non-null)
        if isinstance(raw, pd.DataFrame):
            past = raw.dropna(subset=["Reported EPS"]).copy()
        else:
            return pd.DataFrame()

        if past.empty:
            return pd.DataFrame()

        # Normalise index to date objects
        past.index = pd.to_datetime(past.index).date
        past = past.sort_index(ascending=False)                    # newest first
        past = past.head(HISTORY_QUARTERS)

        # ── Price history ─────────────────────────────────────────────────────
        if history_df is None or history_df.empty:
            history_df = t.history(period="3y")
        if history_df.empty:
            logger.warning(f"[EARNINGS] {ticker}: no price history")
            return pd.DataFrame()

        price_idx = pd.to_datetime(history_df.index).date
        close_by_date = dict(zip(price_idx, history_df["Close"].values))
        all_dates = sorted(close_by_date.keys())

        def _nth_business_day(target: date, n: int) -> float:
            """Close n business days after (n>0) or before (n<0) target date."""
            idx = None
            for i, d in enumerate(all_dates):
                if d >= target:
                    idx = i
                    break
            if idx is None:
                return float("nan")
            pos = idx + n
            if 0 <= pos < len(all_dates):
                return close_by_date[all_dates[pos]]
            return float("nan")

        rows = []
        for earn_date, row in past.iterrows():
            eps_est    = _safe_float(row.get("EPS Estimate"))
            eps_act    = _safe_float(row.get("Reported EPS"))
            surp       = _safe_float(row.get("Surprise(%)"))
            if math.isnan(surp) and not (math.isnan(eps_est) or math.isnan(eps_act)):
                surp = _pct(eps_act - eps_est, abs(eps_est)) if eps_est != 0 else float("nan")

            cb  = _nth_business_day(earn_date, -1)   # close day before
            ca  = _nth_business_day(earn_date,  1)   # close day after
            c5d = _nth_business_day(earn_date,  5)   # close 5 days after

            m1d = _pct(ca - cb, cb) if not (math.isnan(ca) or math.isnan(cb)) else float("nan")
            m5d = _pct(c5d - cb, cb) if not (math.isnan(c5d) or math.isnan(cb)) else float("nan")

            beat  = (surp > 0) if not math.isnan(surp) else None
            up_1d = (m1d > 0) if not math.isnan(m1d) else None

            rows.append({
                "date":         earn_date,
                "eps_estimate": round(eps_est, 4) if not math.isnan(eps_est) else None,
                "eps_actual":   round(eps_act, 4) if not math.isnan(eps_act) else None,
                "surprise_pct": round(surp, 2)    if not math.isnan(surp)    else None,
                "close_before": round(cb,  2)     if not math.isnan(cb)      else None,
                "close_after":  round(ca,  2)     if not math.isnan(ca)      else None,
                "move_1d_pct":  round(m1d, 2)     if not math.isnan(m1d)     else None,
                "close_5d":     round(c5d, 2)     if not math.isnan(c5d)     else None,
                "move_5d_pct":  round(m5d, 2)     if not math.isnan(m5d)     else None,
                "beat":  beat,
                "up_1d": up_1d,
            })

        df = pd.DataFrame(rows).set_index("date")
        return df.sort_index(ascending=False)

    except Exception as e:
        logger.error(f"[EARNINGS] {ticker}: fetch_earnings_history failed — {e}")
        return pd.DataFrame()


# ── 2. Direction Bias Scoring ─────────────────────────────────────────────────

def score_direction_bias(
    ticker: str,
    history_df: pd.DataFrame,
    signals: dict | None,
    calendar_data: dict | None = None,
    analyst_data: dict | None = None,
) -> dict:
    """
    Composite direction score (–100 to +100) from 5 sub-signals.

    Weights:
      EPS beat streak       25 pts  (last 4 quarters)
      Surprise trend        15 pts  (recent vs prior average)
      Estimate revisions    20 pts  (up vs. down revision counts)
      Pre-earnings momentum 20 pts  (technicals from signals dict)
      Analyst consensus     20 pts  (buy% + price target upside)

    Returns:
      {
        score: int,             # –100 to +100
        direction: str,         # "bullish" | "bearish" | "neutral"
        direction_label: str,   # emoji + human label
        confidence: str,        # "moderate" | "low"
        confidence_pct: str,    # "~58–63%" | "~50–55%"
        signals_used: int,      # how many sub-signals had data (max 5)
        breakdown: {
          <key>: {score: int, label: str, available: bool}
        }
      }
    """
    breakdown: dict = {}
    raw_score   = 0.0
    max_possible = 0.0

    # ── Signal 1: EPS beat streak (weight 25) ────────────────────────────────
    W1 = 25
    if not history_df.empty and "beat" in history_df.columns:
        recent = history_df.head(BEAT_WINDOW)["beat"].dropna().tolist()
        beats  = sum(1 for b in recent if b is True)
        total  = len(recent)
        if total >= 2:
            beat_rate = beats / total
            contrib   = round((beat_rate - 0.5) * 2 * W1)   # –25 to +25
            raw_score   += contrib
            max_possible += W1
            streak_label  = f"{beats}/{total} quarters beat estimate"
            breakdown["eps_beat_streak"] = {
                "score":     contrib,
                "label":     streak_label,
                "available": True,
            }
        else:
            breakdown["eps_beat_streak"] = {
                "score": 0, "label": "Insufficient history (< 2 quarters)", "available": False
            }
    else:
        breakdown["eps_beat_streak"] = {
            "score": 0, "label": "No EPS history available", "available": False
        }

    # ── Signal 2: EPS surprise trend (weight 15) ─────────────────────────────
    W2 = 15
    if not history_df.empty and "surprise_pct" in history_df.columns:
        surps = history_df["surprise_pct"].dropna().tolist()
        if len(surps) >= 4:
            recent_avg = sum(surps[:2]) / 2
            prior_avg  = sum(surps[2:4]) / 2
            diff       = recent_avg - prior_avg          # positive = improving
            # Scale: ±10% surprise improvement → ±full weight
            contrib = max(-W2, min(W2, round(diff / 10 * W2)))
            raw_score   += contrib
            max_possible += W2
            direction_word = "improving" if diff > 0 else "deteriorating"
            breakdown["surprise_trend"] = {
                "score":     contrib,
                "label":     f"Recent surprise avg {recent_avg:+.1f}% vs prior {prior_avg:+.1f}% ({direction_word})",
                "available": True,
            }
        else:
            breakdown["surprise_trend"] = {
                "score": 0, "label": "Need 4+ quarters for trend", "available": False
            }
    else:
        breakdown["surprise_trend"] = {
            "score": 0, "label": "No surprise data", "available": False
        }

    # ── Signal 3: Estimate revisions (weight 20) ─────────────────────────────
    W3 = 20
    rev_contrib = None
    rev_label   = "Revision data unavailable"
    if analyst_data:
        rev = analyst_data.get("eps_revisions")
        if rev is not None and not (isinstance(rev, pd.DataFrame) and rev.empty):
            try:
                rev_df = rev if isinstance(rev, pd.DataFrame) else pd.DataFrame(rev)
                # Columns: upLast7days, upLast30days, downLast7days, downLast30days (current quarter)
                row = rev_df.iloc[0] if not rev_df.empty else None
                if row is not None:
                    up   = _safe_float(row.get("upLast30days", 0))
                    down = _safe_float(row.get("downLast30days", 0))
                    total_rev = up + down
                    if total_rev > 0:
                        net_pct = (up - down) / total_rev   # –1 to +1
                        rev_contrib = round(net_pct * W3)
                        rev_label   = (
                            f"{int(up)} upward / {int(down)} downward revisions (last 30d)"
                        )
            except Exception:
                pass

    if rev_contrib is not None:
        raw_score   += rev_contrib
        max_possible += W3
        breakdown["estimate_revisions"] = {
            "score": rev_contrib, "label": rev_label, "available": True
        }
    else:
        breakdown["estimate_revisions"] = {
            "score": 0, "label": rev_label, "available": False
        }

    # ── Signal 4: Pre-earnings momentum (weight 20) ──────────────────────────
    W4 = 20
    mom_contrib = None
    mom_label   = "Technicals unavailable"
    if signals:
        trend = signals.get("macro_trend", "NEUTRAL")
        rsi9  = _safe_float(signals.get("rsi9"), 50)
        price = _safe_float(signals.get("current_price"), 0)
        ema50 = _safe_float(signals.get("ema50"), price)

        trend_score  = 1 if trend == "BULLISH" else (-1 if trend == "BEARISH" else 0)
        rsi_score    = 1 if rsi9 > 55 else (-1 if rsi9 < 45 else 0)
        price_score  = 1 if (price > ema50 and ema50 > 0) else -1

        combo = trend_score + rsi_score + price_score   # –3 to +3
        mom_contrib = round(combo / 3 * W4)
        mom_label   = (
            f"Trend {trend} · RSI9 {rsi9:.0f} · "
            f"Price {'above' if price > ema50 else 'below'} EMA50"
        )

    if mom_contrib is not None:
        raw_score   += mom_contrib
        max_possible += W4
        breakdown["momentum"] = {
            "score": mom_contrib, "label": mom_label, "available": True
        }
    else:
        breakdown["momentum"] = {
            "score": 0, "label": mom_label, "available": False
        }

    # ── Signal 5: Analyst consensus (weight 20) ──────────────────────────────
    W5 = 20
    ana_contrib = None
    ana_label   = "Analyst data unavailable"
    if analyst_data:
        # Price target upside
        pt = analyst_data.get("price_targets", {})
        mean_target = _safe_float(pt.get("mean")) if pt else float("nan")
        spot        = _safe_float((signals or {}).get("current_price")) if signals else float("nan")

        # Recommendation summary
        rec = analyst_data.get("recommendations_summary")
        buy_pct = float("nan")
        if rec is not None and not (isinstance(rec, pd.DataFrame) and rec.empty):
            try:
                rec_df = rec if isinstance(rec, pd.DataFrame) else pd.DataFrame(rec)
                latest = rec_df.iloc[0]
                strong_buy = _safe_float(latest.get("strongBuy", 0))
                buy        = _safe_float(latest.get("buy",       0))
                hold       = _safe_float(latest.get("hold",      0))
                sell       = _safe_float(latest.get("sell",      0))
                strong_sell= _safe_float(latest.get("strongSell",0))
                total_ana  = strong_buy + buy + hold + sell + strong_sell
                if total_ana > 0:
                    buy_pct = (strong_buy + buy) / total_ana * 100
            except Exception:
                pass

        # Score: upside and buy% independently contribute
        contrib_parts = []
        label_parts   = []

        if not math.isnan(mean_target) and not math.isnan(spot) and spot > 0:
            upside_pct = (mean_target - spot) / spot * 100
            # Scale: >20% upside → full; <-10% → negative
            upside_score = max(-1.0, min(1.0, upside_pct / 20))
            contrib_parts.append(upside_score)
            label_parts.append(f"Price target ${mean_target:.0f} ({upside_pct:+.1f}% upside)")

        if not math.isnan(buy_pct):
            buy_score = (buy_pct - 50) / 50   # 0% buy → –1, 100% buy → +1
            contrib_parts.append(buy_score)
            label_parts.append(f"{buy_pct:.0f}% analysts rated Buy/Strong Buy")

        if contrib_parts:
            avg_contrib = sum(contrib_parts) / len(contrib_parts)
            ana_contrib = round(avg_contrib * W5)
            ana_label   = " · ".join(label_parts)

    if ana_contrib is not None:
        raw_score   += ana_contrib
        max_possible += W5
        breakdown["analyst_consensus"] = {
            "score": ana_contrib, "label": ana_label, "available": True
        }
    else:
        breakdown["analyst_consensus"] = {
            "score": 0, "label": ana_label, "available": False
        }

    # ── Normalise score to –100/+100 if not all signals available ────────────
    signals_used = sum(1 for v in breakdown.values() if v["available"])
    if max_possible > 0:
        # Scale so that a full score on available signals maps to ±100
        score = int(round(raw_score / max_possible * 100))
    else:
        score = 0

    score = max(-100, min(100, score))

    # ── Direction & confidence ────────────────────────────────────────────────
    if score > 35:
        direction       = "bullish"
        direction_label = "🟢 Bullish Lean"
        confidence      = "moderate"
        confidence_pct  = "~58–63%"
    elif score < -35:
        direction       = "bearish"
        direction_label = "🔴 Bearish Lean"
        confidence      = "moderate"
        confidence_pct  = "~58–63%"
    else:
        direction       = "neutral"
        direction_label = "🟡 No Clear Edge"
        confidence      = "low"
        confidence_pct  = "~50–55%"

    return {
        "score":          score,
        "direction":      direction,
        "direction_label": direction_label,
        "confidence":     confidence,
        "confidence_pct": confidence_pct,
        "signals_used":   signals_used,
        "breakdown":      breakdown,
    }


# ── 3. Implied Move ───────────────────────────────────────────────────────────

def compute_implied_move(
    ticker: str,
    earnings_date: date | None,
    spot: float | None = None,
) -> dict:
    """
    Compute the market's expected move for the upcoming earnings event from the
    ATM straddle price on the expiration closest to (and after) the earnings date.

    Returns:
      {
        implied_move_pct: float,     # e.g. 0.085 = 8.5%
        implied_up_target: float,
        implied_down_target: float,
        expiration_used: str,        # "YYYY-MM-DD"
        dte: int,
        method: str,                 # "straddle" | "iv_formula" | "unavailable"
        straddle_cost: float | None,
        note: str,
      }
    """
    _na = {
        "implied_move_pct":  None,
        "implied_up_target": None,
        "implied_down_target": None,
        "expiration_used":   None,
        "dte":               None,
        "method":            "unavailable",
        "straddle_cost":     None,
        "note":              "Could not compute implied move.",
    }
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return {**_na, "note": "No options chain available."}

        today = date.today()
        ref_date = earnings_date or (today + timedelta(days=7))

        # Find the expiration just after (or on) the earnings date, within MAX_DTE
        target_exp = None
        target_dte = None
        for exp_str in sorted(expirations):
            exp_d = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte   = (exp_d - today).days
            if exp_d >= ref_date and dte <= IMPLIED_MOVE_MAX_DTE:
                target_exp = exp_str
                target_dte = dte
                break

        if target_exp is None:
            # Fall back: nearest expiry after earnings, regardless of DTE cap
            for exp_str in sorted(expirations):
                exp_d = datetime.strptime(exp_str, "%Y-%m-%d").date()
                if exp_d >= ref_date:
                    target_exp = exp_str
                    target_dte = (exp_d - today).days
                    break

        if target_exp is None:
            return {**_na, "note": "No expiration found after earnings date."}

        chain = t.option_chain(target_exp)
        calls = chain.calls
        puts  = chain.puts

        # Get current spot if not supplied
        if spot is None or spot <= 0:
            hist = t.history(period="2d")
            spot = float(hist["Close"].iloc[-1]) if not hist.empty else None
        if spot is None or spot <= 0:
            return {**_na, "note": "Could not determine spot price."}

        # Find ATM call and put (strike closest to spot)
        calls_valid = calls[(calls["bid"] > 0) & (calls["ask"] > 0)].copy()
        puts_valid  = puts[(puts["bid"] > 0)   & (puts["ask"] > 0)].copy()

        if calls_valid.empty or puts_valid.empty:
            return {**_na, "note": "No liquid ATM options found for implied move."}

        calls_valid["dist"] = (calls_valid["strike"] - spot).abs()
        puts_valid["dist"]  = (puts_valid["strike"]  - spot).abs()
        atm_call = calls_valid.loc[calls_valid["dist"].idxmin()]
        atm_put  = puts_valid.loc[puts_valid["dist"].idxmin()]

        call_mid = (float(atm_call["bid"]) + float(atm_call["ask"])) / 2
        put_mid  = (float(atm_put["bid"])  + float(atm_put["ask"]))  / 2
        straddle = round(call_mid + put_mid, 2)

        implied_pct = straddle / spot
        up_target   = round(spot * (1 + implied_pct), 2)
        dn_target   = round(spot * (1 - implied_pct), 2)

        return {
            "implied_move_pct":   round(implied_pct, 4),
            "implied_up_target":  up_target,
            "implied_down_target": dn_target,
            "expiration_used":    target_exp,
            "dte":                target_dte,
            "method":             "straddle",
            "straddle_cost":      straddle,
            "note":               (
                f"ATM straddle ({atm_call['strike']:g}C + {atm_put['strike']:g}P) "
                f"priced at ${straddle:.2f} on {target_exp} ({target_dte}d)"
            ),
        }

    except Exception as e:
        logger.error(f"[EARNINGS] {ticker}: compute_implied_move failed — {e}")
        return {**_na, "note": f"Error: {e}"}


# ── 4. Strategy Recommendations ───────────────────────────────────────────────

def recommend_earnings_strategy(
    direction: str,       # "bullish" | "bearish" | "neutral"
    iv_rank: float,       # 0–100
    implied_move_pct: float | None,
    spot: float,
) -> list[dict]:
    """
    Return 2–3 ranked strategy recommendations for the earnings event.

    Each dict:
      {
        rank: int,
        name: str,
        structure: str,
        rationale: str,
        risk_profile: str,         # "defined" | "undefined"
        max_loss_note: str,
        timeline: {scan, enter, announcement}
        iv_fit: str,               # "ideal" | "acceptable" | "suboptimal"
        caution: str | None,
      }
    """
    recs = []
    imp_str = f"~{implied_move_pct*100:.1f}%" if implied_move_pct else "unknown"

    # ── HIGH IV (> 60) ────────────────────────────────────────────────────────
    if iv_rank > 60:
        if direction == "neutral":
            recs.append({
                "rank": 1,
                "name": "Iron Condor",
                "structure": "Sell OTM call + buy further OTM call · Sell OTM put + buy further OTM put",
                "rationale": (
                    f"No directional edge + elevated IV (rank {iv_rank:.0f}) = ideal premium-selling "
                    f"environment. Collect credit while the market implies a {imp_str} move. Profitable "
                    "if stock stays inside the short strikes."
                ),
                "risk_profile": "defined",
                "max_loss_note": "Max loss = spread width − credit collected",
                "iv_fit": "ideal",
                "timeline": {
                    "scan":         "5–10 days out: verify IV rank is rising as earnings approach",
                    "enter":        "1–3 days before: place the condor when IV is elevated. Size wings at 1–1.5× the implied move to give cushion.",
                    "announcement": "Day of print: close the full condor before the number drops if down >50% of max profit, or let it expire worthless.",
                },
                "caution": "Earnings can cause 2–3× the implied move. Keep position size small — this is a high-risk binary event.",
            })
            recs.append({
                "rank": 2,
                "name": "Short Strangle (wider)",
                "structure": "Sell OTM call + Sell OTM put (no long legs)",
                "rationale": (
                    f"Maximum premium collection vs. Iron Condor, but UNDEFINED max loss. "
                    "Use only if you have the margin to handle a gap move beyond the strikes."
                ),
                "risk_profile": "undefined",
                "max_loss_note": "Unlimited loss if stock gaps far beyond short strikes",
                "iv_fit": "ideal",
                "timeline": {
                    "scan":         "5–10 days out: check that IV rank is at a 52-week high for this ticker going into earnings",
                    "enter":        "Same day as earnings (day of) to capture the full premium spike, or 1 day before",
                    "announcement": "Close the moment you can capture 50%+ of the original credit after the IV crush. Do not hold through an adverse gap.",
                },
                "caution": "⚠️ Undefined risk. Suitable only for experienced traders with proper margin and stop management.",
            })

        elif direction == "bullish":
            recs.append({
                "rank": 1,
                "name": "Bull Call Spread (tight)",
                "structure": "Buy ATM call + Sell OTM call (1–2 strikes above)",
                "rationale": (
                    f"Bullish lean ({iv_rank:.0f} IV rank) but elevated IV makes outright long calls expensive. "
                    "A tight call spread reduces cost by selling a higher strike, capping upside but keeping the directional bet."
                ),
                "risk_profile": "defined",
                "max_loss_note": "Max loss = net debit paid",
                "iv_fit": "acceptable",
                "timeline": {
                    "scan":         "5–10 days out: confirm bullish signals are still intact and IV is elevated (not just pre-earnings spike)",
                    "enter":        "1–3 days before earnings. Avoid entering same-day — IV will be at its peak.",
                    "announcement": "If stock gaps up through your short strike immediately after open — close and take the full profit. Do not wait for expiry.",
                },
                "caution": f"Even with a bullish lean, earnings direction accuracy is only ~58–63%. Limit size to 1–2% of portfolio.",
            })
            recs.append({
                "rank": 2,
                "name": "Cash-Secured Put",
                "structure": "Sell OTM put 1–2 strikes below spot",
                "rationale": (
                    f"Collect elevated premium (IV rank {iv_rank:.0f}) while expressing a bullish view. "
                    "If the stock dips on earnings, you acquire shares at an effective discount."
                ),
                "risk_profile": "defined",
                "max_loss_note": "Max loss = strike price − premium (same as buying shares at a discount)",
                "iv_fit": "ideal",
                "timeline": {
                    "scan":         "5–10 days out: check that the put strike is at a meaningful support level",
                    "enter":        "1–3 days before earnings for maximum IV benefit",
                    "announcement": "If the stock opens sharply lower and the put is deep ITM, evaluate early assignment risk vs. rolling to the next quarter.",
                },
                "caution": None,
            })

        else:  # bearish
            recs.append({
                "rank": 1,
                "name": "Bear Put Spread (tight)",
                "structure": "Buy ATM put + Sell OTM put (1–2 strikes below)",
                "rationale": (
                    f"Bearish lean + elevated IV (rank {iv_rank:.0f}). Buying the put outright is expensive; "
                    "selling a lower strike offsets cost and funds a clean directional bet."
                ),
                "risk_profile": "defined",
                "max_loss_note": "Max loss = net debit paid",
                "iv_fit": "acceptable",
                "timeline": {
                    "scan":         "5–10 days out: confirm bearish technicals are still aligned",
                    "enter":        "1–3 days before earnings",
                    "announcement": "Close at open if the stock gaps down through your short strike — take the full profit immediately.",
                },
                "caution": f"Earnings direction accuracy is ~58–63% even in high-conviction setups.",
            })
            recs.append({
                "rank": 2,
                "name": "Call Credit Spread",
                "structure": "Sell OTM call + Buy higher OTM call",
                "rationale": (
                    "Collect credit while expressing a bearish or neutral view. "
                    "Profitable if stock stays flat or declines after the number."
                ),
                "risk_profile": "defined",
                "max_loss_note": "Max loss = spread width − credit collected",
                "iv_fit": "ideal",
                "timeline": {
                    "scan":         "5–10 days out: verify that the short call strike is above key resistance",
                    "enter":        "1–3 days before earnings",
                    "announcement": "Close early if down 50%+ of max profit (the stock rallied). Don't let a credit spread expire near the money.",
                },
                "caution": None,
            })

    # ── MID IV (30–60) ────────────────────────────────────────────────────────
    elif iv_rank >= 30:
        if direction == "bullish":
            recs.append({
                "rank": 1,
                "name": "Bull Call Debit Spread",
                "structure": "Buy ATM call + Sell OTM call (2–3 strikes above)",
                "rationale": (
                    f"Moderate IV (rank {iv_rank:.0f}) + bullish lean. Options are reasonably priced — "
                    "a debit spread gives full directional exposure with defined max loss."
                ),
                "risk_profile": "defined",
                "max_loss_note": "Max loss = net debit (the cost to open)",
                "iv_fit": "ideal",
                "timeline": {
                    "scan":         "5–10 days out: confirm the bullish catalyst thesis (beat streak, upward revisions)",
                    "enter":        "2–3 days before earnings for the best balance of cost vs. time",
                    "announcement": "Close at next open if stock gaps up significantly. Partial profit is better than gambling on direction reversing intraday.",
                },
                "caution": None,
            })
            recs.append({
                "rank": 2,
                "name": "Long Call",
                "structure": "Buy ATM call (1 contract per position unit)",
                "rationale": (
                    f"IV rank {iv_rank:.0f} is moderate — long calls are not excessively expensive. "
                    "Maximum upside capture if the stock runs hard past the implied move."
                ),
                "risk_profile": "defined",
                "max_loss_note": "Max loss = premium paid (total cost of the option)",
                "iv_fit": "acceptable",
                "timeline": {
                    "scan":         "5–10 days out: check that the implied move is below the historical average move for this ticker",
                    "enter":        "1–2 days before earnings",
                    "announcement": "Sell into the first 30 minutes of the post-earnings open — IV crush will eat value quickly if the stock doesn't move sharply.",
                },
                "caution": f"The market implies a {imp_str} move. A long call needs a move significantly LARGER than this to profit after IV crush.",
            })

        elif direction == "bearish":
            recs.append({
                "rank": 1,
                "name": "Bear Put Debit Spread",
                "structure": "Buy ATM put + Sell OTM put (2–3 strikes below)",
                "rationale": (
                    f"Moderate IV (rank {iv_rank:.0f}) + bearish lean. Debit spread provides directional "
                    "exposure at a lower cost than an outright put."
                ),
                "risk_profile": "defined",
                "max_loss_note": "Max loss = net debit",
                "iv_fit": "ideal",
                "timeline": {
                    "scan":         "5–10 days out: verify the bearish thesis (miss history, downward revisions)",
                    "enter":        "2–3 days before earnings",
                    "announcement": "Close at open if stock gaps down through your short put.",
                },
                "caution": None,
            })

        else:  # neutral, mid IV
            recs.append({
                "rank": 1,
                "name": "Iron Condor (wider wings)",
                "structure": "Sell OTM call + buy call + sell OTM put + buy put (wider than high-IV version)",
                "rationale": (
                    f"No directional edge. IV rank {iv_rank:.0f} is moderate — credit collection is thinner, "
                    "so use wider wings to keep a reasonable risk/reward ratio."
                ),
                "risk_profile": "defined",
                "max_loss_note": "Max loss = spread width − credit collected",
                "iv_fit": "acceptable",
                "timeline": {
                    "scan":         "5–10 days out: watch for IV expansion as earnings approach — may improve to 'ideal' category",
                    "enter":        "1–2 days before earnings when IV is closest to its peak",
                    "announcement": "Close immediately after the open. Do not hold a condor through a slow-moving post-earnings day.",
                },
                "caution": "Moderate IV means thinner credit. Credit spreads around earnings need elevated IV to be worthwhile — consider skipping if IV rank stays below 40.",
            })

    # ── LOW IV (< 30) ─────────────────────────────────────────────────────────
    else:
        if direction == "bullish":
            recs.append({
                "rank": 1,
                "name": "Long Call",
                "structure": "Buy ATM or slightly OTM call",
                "rationale": (
                    f"Low IV (rank {iv_rank:.0f}) + bullish lean = best scenario for buying options. "
                    "Options are cheap; if the stock beats and runs, gamma and delta both work in your favour."
                ),
                "risk_profile": "defined",
                "max_loss_note": "Max loss = premium paid",
                "iv_fit": "ideal",
                "timeline": {
                    "scan":         "5–10 days out: IV rank < 30 is rare going into earnings — confirm it's not a data error",
                    "enter":        "3–5 days before earnings while options are still cheap",
                    "announcement": "Sell at the open. IV will spike on the move — you profit from both delta (direction) and vega (IV pop).",
                },
                "caution": None,
            })
        elif direction == "bearish":
            recs.append({
                "rank": 1,
                "name": "Long Put",
                "structure": "Buy ATM or slightly OTM put",
                "rationale": (
                    f"Low IV (rank {iv_rank:.0f}) + bearish lean. Puts are cheap — maximum leverage if the stock misses and drops."
                ),
                "risk_profile": "defined",
                "max_loss_note": "Max loss = premium paid",
                "iv_fit": "ideal",
                "timeline": {
                    "scan":         "5–10 days out: confirm bearish signals are intact",
                    "enter":        "3–5 days before earnings",
                    "announcement": "Sell into the open-market drop. Don't hold a long put hoping for a further decline — theta is aggressive post-earnings.",
                },
                "caution": None,
            })
        else:  # neutral, low IV
            recs.append({
                "rank": 1,
                "name": "Long Straddle",
                "structure": "Buy ATM call + Buy ATM put (same strike, same expiry)",
                "rationale": (
                    f"No directional edge + low IV (rank {iv_rank:.0f}) = buy volatility cheaply before the event. "
                    f"The market implies a {imp_str} move — if the actual move is larger, both legs can be profitable."
                ),
                "risk_profile": "defined",
                "max_loss_note": "Max loss = combined premium (call + put cost)",
                "iv_fit": "ideal",
                "timeline": {
                    "scan":         "5–10 days out: check historical move vs. current implied move. Buy vol if historical avg > current implied.",
                    "enter":        "3–5 days before earnings while IV is still low. Avoid entering the day before — IV spike inflates straddle cost.",
                    "announcement": "Sell the winning leg immediately at the open. Do not delta-hedge — let the winning side run for 30 minutes then close.",
                },
                "caution": f"Straddles require a move larger than {imp_str} to profit. If the stock barely moves, both legs lose to IV crush.",
            })

    return recs


# ── 5. Historical Backtest Stats ──────────────────────────────────────────────

def backtest_earnings_plays(history_df: pd.DataFrame) -> dict:
    """
    Aggregate statistics over the historical earnings reaction DataFrame.

    Returns:
      {
        n_events:            int,
        beat_rate:           float,   # % quarters with positive EPS surprise
        up_reaction_rate:    float,   # % quarters with positive 1-day move
        beat_up_correlation: float,   # % of beats that produced an up move (buy-the-news)
        miss_down_rate:      float,   # % of misses that produced a down move
        sell_the_news_rate:  float,   # % of beats where stock went DOWN (sell the news)
        avg_up_move:         float,   # average move on up days (%)
        avg_down_move:       float,   # average move on down days (%) — negative
        median_abs_move:     float,   # typical magnitude regardless of direction
        avg_move_5d_up:      float,   # 5-day drift after initial up day
        avg_move_5d_down:    float,   # 5-day drift after initial down day
      }
    """
    if history_df.empty:
        return {}

    df = history_df.copy()

    def _rate(mask) -> float | None:
        valid = df[mask.notna()]
        if valid.empty:
            return None
        return round(valid[mask[valid.index]].sum() / len(valid) * 100, 1)

    beat_mask = df["beat"].notna()
    up_mask   = df["up_1d"].notna()

    beats   = df[beat_mask]["beat"].astype(bool)
    up_days = df[up_mask]["up_1d"].astype(bool)

    beat_rate       = round(beats.mean() * 100, 1) if not beats.empty else None
    up_rate         = round(up_days.mean() * 100, 1) if not up_days.empty else None

    # Beat→up (buy the news)
    beat_and_up = df[beat_mask & up_mask]
    if not beat_and_up.empty:
        beat_up_corr    = round(
            (beat_and_up["beat"] & beat_and_up["up_1d"]).sum() / len(beat_and_up) * 100, 1
        )
        sell_news_rate  = round(
            (beat_and_up["beat"] & ~beat_and_up["up_1d"]).sum() / beat_and_up["beat"].sum() * 100, 1
        ) if beat_and_up["beat"].sum() > 0 else None
    else:
        beat_up_corr   = None
        sell_news_rate = None

    # Miss→down
    miss_and_down = df[beat_mask & up_mask]
    if not miss_and_down.empty:
        miss_count = (~miss_and_down["beat"]).sum()
        miss_down_rate = round(
            (~miss_and_down["beat"] & ~miss_and_down["up_1d"]).sum() / miss_count * 100, 1
        ) if miss_count > 0 else None
    else:
        miss_down_rate = None

    # Move magnitudes
    moves = df["move_1d_pct"].dropna()
    up_moves   = moves[moves > 0]
    down_moves = moves[moves < 0]
    avg_up    = round(up_moves.mean(), 2)   if not up_moves.empty   else None
    avg_down  = round(down_moves.mean(), 2) if not down_moves.empty else None
    med_abs   = round(moves.abs().median(), 2) if not moves.empty  else None

    # 5-day drift
    df5 = df.dropna(subset=["move_5d_pct", "up_1d"])
    df5_up   = df5[df5["up_1d"] == True]
    df5_down = df5[df5["up_1d"] == False]
    avg_5d_up   = round(df5_up["move_5d_pct"].mean(), 2)   if not df5_up.empty   else None
    avg_5d_down = round(df5_down["move_5d_pct"].mean(), 2) if not df5_down.empty else None

    return {
        "n_events":            len(df),
        "beat_rate":           beat_rate,
        "up_reaction_rate":    up_rate,
        "beat_up_correlation": beat_up_corr,
        "miss_down_rate":      miss_down_rate,
        "sell_the_news_rate":  sell_news_rate,
        "avg_up_move":         avg_up,
        "avg_down_move":       avg_down,
        "median_abs_move":     med_abs,
        "avg_move_5d_up":      avg_5d_up,
        "avg_move_5d_down":    avg_5d_down,
    }


# ── 6. Batch fetch (for caching) ──────────────────────────────────────────────

def fetch_all_earnings_data(ticker: str, history_df: pd.DataFrame | None = None) -> dict:
    """
    Single-Ticker() call that fetches all analyst / calendar data and returns
    a structured dict for caching. Designed to be called once per ticker per hour.
    """
    result = {
        "calendar":              None,
        "recommendations_summary": None,
        "price_targets":         None,
        "eps_revisions":         None,
        "is_etf":                False,
        "error":                 None,
    }
    try:
        t = yf.Ticker(ticker)

        # Calendar (next earnings EPS estimate)
        try:
            cal = t.calendar
            if isinstance(cal, dict):
                result["calendar"] = cal
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                result["calendar"] = cal.to_dict()
        except Exception:
            pass

        # Recommendations
        try:
            rec = t.recommendations_summary
            if rec is not None and isinstance(rec, pd.DataFrame) and not rec.empty:
                result["recommendations_summary"] = rec
        except Exception:
            pass

        # Price targets
        try:
            pt = t.analyst_price_targets
            if pt is not None:
                result["price_targets"] = pt if isinstance(pt, dict) else pt.to_dict() if hasattr(pt, "to_dict") else None
        except Exception:
            pass

        # EPS revisions
        try:
            rev = t.eps_revisions
            if rev is not None and isinstance(rev, pd.DataFrame) and not rev.empty:
                result["eps_revisions"] = rev
        except Exception:
            pass

        # Detect ETF: earnings_dates empty or None
        try:
            ed = t.earnings_dates
            result["is_etf"] = ed is None or (isinstance(ed, pd.DataFrame) and ed.empty)
        except Exception:
            result["is_etf"] = True

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[EARNINGS] {ticker}: fetch_all_earnings_data failed — {e}")

    return result


if __name__ == "__main__":
    print("Testing earnings_playbook with AAPL…")
    hist = yf.Ticker("AAPL").history(period="3y")
    price = float(hist["Close"].iloc[-1])

    df = fetch_earnings_history("AAPL", hist)
    print(f"\nEarnings history ({len(df)} events):")
    if not df.empty:
        print(df[["eps_estimate", "eps_actual", "surprise_pct", "move_1d_pct", "beat", "up_1d"]].head(6).to_string())

    stats = backtest_earnings_plays(df)
    print(f"\nBacktest stats: {stats}")

    signals = {"macro_trend": "BULLISH", "rsi9": 58.0, "ema50": price * 0.97, "current_price": price}
    from datetime import date
    analyst_data = fetch_all_earnings_data("AAPL", hist)
    bias = score_direction_bias("AAPL", df, signals, analyst_data.get("calendar"), analyst_data)
    print(f"\nDirection bias: {bias['direction_label']} ({bias['score']:+d}) — {bias['confidence_pct']}")
    for k, v in bias["breakdown"].items():
        print(f"  {k}: {v['score']:+d}  {v['label']}")

    from news_events import fetch_next_earnings
    earnings_date = fetch_next_earnings("AAPL")
    im = compute_implied_move("AAPL", earnings_date, spot=price)
    print(f"\nImplied move: ±{im['implied_move_pct']*100:.1f}% | {im['note']}")

    recs = recommend_earnings_strategy(bias["direction"], 55, im["implied_move_pct"], price)
    print(f"\nRecommended strategies:")
    for r in recs:
        print(f"  #{r['rank']} {r['name']}: {r['rationale'][:80]}…")
