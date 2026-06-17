import logging

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)

MIN_HISTORY_ROWS = 200


class TechnicalAnalyzer:
    def __init__(self, df: pd.DataFrame):
        if len(df) < MIN_HISTORY_ROWS:
            raise ValueError(
                f"Insufficient history: {len(df)} rows (need {MIN_HISTORY_ROWS})"
            )
        self.df = df.copy()
        self.close = self.df["Close"]

    def compute_ema(self, window: int) -> pd.Series:
        return ta.trend.EMAIndicator(close=self.close, window=window).ema_indicator()

    def compute_rsi(self, window: int) -> float:
        return float(
            ta.momentum.RSIIndicator(close=self.close, window=window).rsi().iloc[-1]
        )

    def compute_bollinger_bands(self, window: int = 20, window_dev: int = 2) -> dict:
        bb = ta.volatility.BollingerBands(
            close=self.close, window=window, window_dev=window_dev
        )
        upper = float(bb.bollinger_hband().iloc[-1])
        lower = float(bb.bollinger_lband().iloc[-1])
        mid = float(bb.bollinger_mavg().iloc[-1])
        last_close = float(self.close.iloc[-1])
        return {
            "upper": upper,
            "lower": lower,
            "mid": mid,
            "upper_touch": last_close >= upper,
            "lower_touch": last_close <= lower,
        }

    def compute_hv_and_iv_rank(self, current_atm_iv: float | None = None) -> dict:
        """
        current_atm_iv: ATM implied vol from the options chain as a decimal (e.g. 0.35).
        Falls back to 21-day HV if None.
        """
        log_returns = np.log(self.close / self.close.shift(1))
        hv_series = log_returns.rolling(21).std() * np.sqrt(252)
        hv_252_min = float(hv_series.min())
        hv_252_max = float(hv_series.max())
        hv_current = float(hv_series.iloc[-1])

        current_iv = current_atm_iv if (current_atm_iv is not None and current_atm_iv > 0) \
                     else hv_current

        if hv_252_max == hv_252_min:
            iv_rank = 50.0
        else:
            iv_rank = (current_iv - hv_252_min) / (hv_252_max - hv_252_min) * 100.0
            iv_rank = max(0.0, min(100.0, iv_rank))

        return {
            "hv_current": round(hv_current, 4),
            "hv_252_min": round(hv_252_min, 4),
            "hv_252_max": round(hv_252_max, 4),
            "iv_rank": round(iv_rank, 2),
            "high_vol_env": iv_rank > 45.0,
        }

    def compute_macro_trend(self, ema50: float, ema200: float) -> str:
        threshold = ema200 * 0.005
        if ema50 > ema200 + threshold:
            return "BULLISH"
        if ema50 < ema200 - threshold:
            return "BEARISH"
        return "NEUTRAL"

    def run_all(self, current_atm_iv: float | None = None) -> dict | None:
        try:
            ema50 = float(self.compute_ema(50).iloc[-1])
            ema200 = float(self.compute_ema(200).iloc[-1])
            macro_trend = self.compute_macro_trend(ema50, ema200)

            rsi9 = self.compute_rsi(9)
            rsi14 = self.compute_rsi(14)
            bb = self.compute_bollinger_bands()
            hv_iv = self.compute_hv_and_iv_rank(current_atm_iv)

            return {
                "macro_trend": macro_trend,
                "ema50": round(ema50, 2),
                "ema200": round(ema200, 2),
                "rsi9": round(rsi9, 2),
                "rsi14": round(rsi14, 2),
                "rsi9_overbought": rsi9 > 70,
                "rsi9_oversold": rsi9 < 30,
                "bb_upper": round(bb["upper"], 2),
                "bb_lower": round(bb["lower"], 2),
                "bb_mid": round(bb["mid"], 2),
                "bb_upper_touch": bb["upper_touch"],
                "bb_lower_touch": bb["lower_touch"],
                "hv_current": hv_iv["hv_current"],
                "iv_rank": hv_iv["iv_rank"],
                "high_vol_env": hv_iv["high_vol_env"],
                "current_price": round(float(self.close.iloc[-1]), 2),
            }
        except Exception as e:
            logger.error(f"TechnicalAnalyzer.run_all failed: {e}")
            return None


def _price_near_ema50(s: dict, pct: float = 0.03) -> bool:
    """True when price is coiling within pct of EMA-50 (pinning behaviour)."""
    ema50 = s.get("ema50") or s["current_price"]
    if ema50 <= 0:
        return False
    return abs(s["current_price"] - ema50) / ema50 <= pct


def _no_bb_touch(s: dict) -> bool:
    return not s["bb_upper_touch"] and not s["bb_lower_touch"]


# Single source of truth for every strategy's technical trigger.
# Used by has_any_trigger() (Phase-1 gate) AND by StrategySelector (Phase-2),
# so the two can never drift apart.
# IRON_CONDOR has no standalone trigger — it derives from CCS + PCS both qualifying.
STRATEGY_TRIGGERS: dict = {
    # ── Credit spreads ────────────────────────────────────────────────────────
    "CALL_CREDIT_SPREAD": lambda s: (
        s["macro_trend"] in ("BEARISH", "NEUTRAL")
        or (s["macro_trend"] == "BULLISH" and s["rsi9"] > 72
            and s["bb_upper_touch"] and s["high_vol_env"])
    ),
    "PUT_CREDIT_SPREAD": lambda s: (
        s["macro_trend"] == "BULLISH" and s["rsi9"] < 28
        and s["bb_lower_touch"] and s["high_vol_env"]
    ),
    # Pin requirement (_price_near_ema50): butterflies profit only if price
    # STOPS at the body — a trending stock with neutral RSI must not qualify.
    "IRON_BUTTERFLY": lambda s: (
        40 <= s["rsi9"] <= 60 and s["high_vol_env"] and _no_bb_touch(s)
        and _price_near_ema50(s)
    ),
    # ── Premium income ────────────────────────────────────────────────────────
    "CASH_SECURED_PUT": lambda s: (
        s["macro_trend"] in ("BULLISH", "NEUTRAL") and s["iv_rank"] > 45
    ),
    "COVERED_CALL": lambda s: (
        s["macro_trend"] in ("BULLISH", "NEUTRAL")
        and s["iv_rank"] > 40 and s["rsi9"] <= 70
    ),
    # Same pin requirement as the butterfly — doubly important here because
    # the short strangle's risk is undefined if the stock keeps trending.
    "SHORT_STRANGLE": lambda s: (
        40 <= s["rsi9"] <= 60 and s["high_vol_env"] and _no_bb_touch(s)
        and _price_near_ema50(s)
    ),
    # ── Diagonals (calendar + vertical hybrid: long back-month ITM, short
    #    front-month OTM — income with a directional tilt) ──────────────────────
    "CALL_DIAGONAL_SPREAD": lambda s: (
        s["macro_trend"] == "BULLISH" and 45 <= s["rsi9"] <= 70 and s["iv_rank"] < 60
    ),
    "PUT_DIAGONAL_SPREAD": lambda s: (
        s["macro_trend"] == "BEARISH" and 30 <= s["rsi9"] <= 55 and s["iv_rank"] < 60
    ),
    # ── Calendars (long vega — want low-to-moderate IV at entry) ─────────────
    "CALL_CALENDAR_SPREAD": lambda s: (
        s["macro_trend"] in ("BULLISH", "NEUTRAL")
        and 38 <= s["rsi9"] <= 62 and 15 <= s["iv_rank"] <= 40
        and _no_bb_touch(s) and _price_near_ema50(s)
    ),
    "PUT_CALENDAR_SPREAD": lambda s: (
        s["macro_trend"] in ("BEARISH", "NEUTRAL")
        and 38 <= s["rsi9"] <= 62 and 15 <= s["iv_rank"] <= 40
        and _no_bb_touch(s) and _price_near_ema50(s)
    ),
    # ── Directional debit ─────────────────────────────────────────────────────
    "BULL_CALL_DEBIT_SPREAD": lambda s: (
        s["macro_trend"] == "BULLISH" and 50 <= s["rsi9"] <= 75 and s["iv_rank"] < 50
    ),
    "BEAR_PUT_DEBIT_SPREAD": lambda s: (
        s["macro_trend"] in ("BEARISH", "NEUTRAL")
        and 25 <= s["rsi9"] <= 50 and s["iv_rank"] < 50
    ),
    "LONG_CALL": lambda s: (
        s["macro_trend"] == "BULLISH" and 55 <= s["rsi9"] <= 75 and s["iv_rank"] < 40
    ),
    "LONG_PUT": lambda s: (
        s["macro_trend"] == "BEARISH" and 25 <= s["rsi9"] <= 45 and s["iv_rank"] < 40
    ),
    # ── Long volatility (cheap IV + coiled price = breakout candidates) ──────
    "LONG_STRADDLE": lambda s: (
        s["iv_rank"] < 25 and 40 <= s["rsi9"] <= 60 and _no_bb_touch(s)
    ),
    "LONG_STRANGLE": lambda s: (
        s["iv_rank"] < 25 and 40 <= s["rsi9"] <= 60 and _no_bb_touch(s)
    ),
}


def has_any_trigger(signals: dict) -> bool:
    """True if any strategy trigger condition is met (gates the Phase-2 options fetch)."""
    return any(trigger(signals) for trigger in STRATEGY_TRIGGERS.values())


def analyze_ticker(
    ticker: str, history_df: pd.DataFrame, current_atm_iv: float | None = None
) -> dict | None:
    try:
        return TechnicalAnalyzer(history_df).run_all(current_atm_iv)
    except ValueError as e:
        logger.warning(f"[SKIP] {ticker}: {e}")
        return None
    except Exception as e:
        logger.error(f"[ERROR] {ticker} analyze_ticker: {e}")
        return None


def check_phase1_trigger(signals: dict | None) -> bool:
    if not signals:
        return False
    return has_any_trigger(signals)


if __name__ == "__main__":
    import yfinance as yf

    df = yf.Ticker("SPY").history(period="1y")
    signals = analyze_ticker("SPY", df)
    if signals:
        print("SPY signals:")
        for k, v in signals.items():
            print(f"  {k}: {v}")
        print(f"\n  Triggered: {check_phase1_trigger(signals)}")
