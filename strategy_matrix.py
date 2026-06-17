import logging

import numpy as np
import pandas as pd

from data_fetcher import INDEX_ETFS
from indicators import analyze_ticker, STRATEGY_TRIGGERS

logger = logging.getLogger(__name__)

# Delta target range for short strikes
DELTA_LOW = 0.15
DELTA_HIGH = 0.20
DELTA_TARGET = 0.175

# Spread width search range (points)
SPREAD_WIDTH_RANGE = range(1, 6)

# Liquidity gates
MIN_OI = 1_000
MIN_CREDIT_PCT = 0.20

# Executability gates — keep strikes realistic so fills actually happen:
# a leg quoted under $0.05 bid is effectively dead, and a short strike more
# than 15% from spot at 30-45 DTE almost never offers tradeable premium.
MIN_LEG_BID = 0.05
MAX_STRIKE_DIST = 0.15

# ── Strategy registry ─────────────────────────────────────────────────────────
# Central metadata used by the UI (names, colours, tabs, filters) and by the
# confidence scorer (bias + IV preference). Adding a strategy = one entry here
# + one evaluate method + one trigger in indicators.STRATEGY_TRIGGERS.
#   type:    credit = you collect premium · debit = you pay premium
#   bias:    directional thesis the setup expresses
#   iv_pref: IV environment in which the strategy has edge
STRATEGY_META: dict[str, dict] = {
    "CALL_CREDIT_SPREAD":     {"name": "Call Credit Spread",    "emoji": "🔴", "color": "#f8d7da", "type": "credit", "bias": "bearish",    "iv_pref": "high"},
    "PUT_CREDIT_SPREAD":      {"name": "Put Credit Spread",     "emoji": "🟢", "color": "#d4edda", "type": "credit", "bias": "bullish",    "iv_pref": "high"},
    "IRON_CONDOR":            {"name": "Iron Condor",           "emoji": "🟡", "color": "#fff3cd", "type": "credit", "bias": "neutral",    "iv_pref": "high"},
    "IRON_BUTTERFLY":         {"name": "Iron Butterfly",        "emoji": "🦋", "color": "#ffe9a8", "type": "credit", "bias": "neutral",    "iv_pref": "high"},
    "CASH_SECURED_PUT":       {"name": "Cash-Secured Put",      "emoji": "💵", "color": "#d1f2eb", "type": "credit", "bias": "bullish",    "iv_pref": "high"},
    "COVERED_CALL":           {"name": "Covered Call",          "emoji": "🏦", "color": "#e8f6df", "type": "credit", "bias": "neutral",    "iv_pref": "high"},
    "SHORT_STRANGLE":         {"name": "Short Strangle",        "emoji": "🃏", "color": "#fbd9c9", "type": "credit", "bias": "neutral",    "iv_pref": "high", "undefined_risk": True},
    "CALL_DIAGONAL_SPREAD":   {"name": "Call Diagonal (PMCC)",  "emoji": "📐", "color": "#d9f0f7", "type": "debit",  "bias": "bullish",    "iv_pref": "mid"},
    "PUT_DIAGONAL_SPREAD":    {"name": "Put Diagonal",          "emoji": "🔻", "color": "#f7e3d9", "type": "debit",  "bias": "bearish",    "iv_pref": "mid"},
    "CALL_CALENDAR_SPREAD":   {"name": "Call Calendar",         "emoji": "🔵", "color": "#dce8f8", "type": "debit",  "bias": "neutral",    "iv_pref": "mid"},
    "PUT_CALENDAR_SPREAD":    {"name": "Put Calendar",          "emoji": "🌊", "color": "#cfdef3", "type": "debit",  "bias": "neutral",    "iv_pref": "mid"},
    "BULL_CALL_DEBIT_SPREAD": {"name": "Bull Call Debit",       "emoji": "🟠", "color": "#fff0dc", "type": "debit",  "bias": "bullish",    "iv_pref": "low"},
    "BEAR_PUT_DEBIT_SPREAD":  {"name": "Bear Put Debit",        "emoji": "🟣", "color": "#f0e6ff", "type": "debit",  "bias": "bearish",    "iv_pref": "low"},
    "LONG_CALL":              {"name": "Long Call",             "emoji": "🚀", "color": "#dff6e8", "type": "debit",  "bias": "bullish",    "iv_pref": "low"},
    "LONG_PUT":               {"name": "Long Put",              "emoji": "🪂", "color": "#fde2e4", "type": "debit",  "bias": "bearish",    "iv_pref": "low"},
    "LONG_STRADDLE":          {"name": "Long Straddle",         "emoji": "🎯", "color": "#e2e3f3", "type": "debit",  "bias": "volatility", "iv_pref": "low"},
    "LONG_STRANGLE":          {"name": "Long Strangle",         "emoji": "🎪", "color": "#e7dff6", "type": "debit",  "bias": "volatility", "iv_pref": "low"},
}

# Only vertical credit structures have a meaningful credit-to-width ratio;
# the sidebar Credit% filter applies to these alone.
CREDIT_PCT_STRATEGIES = frozenset(
    {"CALL_CREDIT_SPREAD", "PUT_CREDIT_SPREAD", "IRON_CONDOR", "IRON_BUTTERFLY"}
)
DEBIT_STRATEGIES = frozenset(k for k, v in STRATEGY_META.items() if v["type"] == "debit")

# Optional imports with graceful fallback
try:
    import mibian
    _HAS_MIBIAN = True
except ImportError:
    _HAS_MIBIAN = False
    logger.warning("mibian not installed; falling back to py_vollib / moneyness proxy")

try:
    from py_vollib.black_scholes.greeks.analytical import delta as _pvl_delta
    _HAS_PYVOLLIB = True
except ImportError:
    _HAS_PYVOLLIB = False
    logger.warning("py_vollib not installed; falling back to moneyness proxy for delta")


# ── Delta approximation ───────────────────────────────────────────────────────

def _moneyness_delta(flag: str, S: float, K: float) -> float:
    m = S / K if K > 0 else 1.0
    if flag == "c":
        if m > 1.05:
            return 0.80
        if m > 1.00:
            return 0.55
        if m > 0.97:
            return 0.40
        if m > 0.93:
            return 0.20
        return 0.10
    # Put: negative mirror of call delta (approximate put-call parity)
    return -_moneyness_delta("c", S, K)


def get_delta(
    flag: str,
    S: float,
    K: float,
    iv_pct: float | None,
    dte: int,
    r: float = 0.05,
) -> float:
    """Black-Scholes delta with three-tier fallback: mibian → py_vollib → moneyness."""
    has_iv = iv_pct is not None and not np.isnan(iv_pct) and iv_pct > 0
    valid_dte = dte > 0

    if _HAS_MIBIAN and has_iv and valid_dte:
        try:
            c = mibian.BS([S, K, r * 100, dte], volatility=iv_pct)
            return float(c.callDelta if flag == "c" else c.putDelta)
        except Exception:
            pass

    if _HAS_PYVOLLIB and has_iv and valid_dte:
        try:
            return float(_pvl_delta(flag, S, K, dte / 365.0, r, iv_pct / 100.0))
        except Exception:
            pass

    return _moneyness_delta(flag, S, K)


# ── Liquidity gates ───────────────────────────────────────────────────────────

def passes_bidask_gate(leg: pd.Series, ticker: str) -> bool:
    bid = leg.get("bid", np.nan)
    ask = leg.get("ask", np.nan)
    if pd.isna(bid) or pd.isna(ask) or ask <= 0:
        return False
    mid = (float(bid) + float(ask)) / 2.0
    if mid <= 0:
        return False
    if ticker in INDEX_ETFS:
        return (float(ask) - float(bid)) <= 0.05
    return (float(ask) - float(bid)) / mid <= 0.10


def passes_oi_gate(leg: pd.Series, min_oi: int = MIN_OI) -> bool:
    oi = leg.get("openInterest", 0)
    if pd.isna(oi):
        return False
    return int(oi) >= min_oi


def passes_yield_gate(
    credit: float, spread_width: float, min_pct: float = MIN_CREDIT_PCT
) -> bool:
    return credit >= min_pct * spread_width


# ── Strategy selector ─────────────────────────────────────────────────────────

class StrategySelector:
    def __init__(
        self,
        ticker: str,
        signals: dict,
        options_data: dict | None,
        min_credit_pct: float = MIN_CREDIT_PCT,
        r: float = 0.05,
    ):
        self.ticker = ticker
        self.signals = signals
        self.options_data = options_data
        self.min_credit_pct = min_credit_pct
        self.r = r
        self.S = float(signals["current_price"])

    def _select_expiration_window(self, chain_df: pd.DataFrame) -> pd.DataFrame:
        """
        Target: 30–45 DTE for credit spreads.
        Fallback: if no standard monthly falls in that window (e.g. holiday-shifted
        expiry lands at 28–29 DTE, or the next one jumps to 55+ DTE), pick the
        single monthly expiration closest to 37 DTE within a [21, 60] search range.
        """
        if chain_df is None or chain_df.empty:
            return pd.DataFrame()

        ideal = chain_df[(chain_df["dte"] >= 30) & (chain_df["dte"] <= 45)]
        if not ideal.empty:
            return ideal.reset_index(drop=True)

        # No expiry in ideal window — fall back to nearest monthly in [21, 60]
        fallback = chain_df[(chain_df["dte"] >= 21) & (chain_df["dte"] <= 60)]
        if fallback.empty:
            return pd.DataFrame()

        # Among available expirations, pick the one whose DTE is closest to 37
        exp_dtes = fallback.groupby("expiration")["dte"].first()
        best_exp = (exp_dtes - 37).abs().idxmin()
        return fallback[fallback["expiration"] == best_exp].reset_index(drop=True)

    def _find_short_leg(
        self,
        chain_df: pd.DataFrame,
        flag: str,
        delta_low: float = DELTA_LOW,
        delta_high: float = DELTA_HIGH,
        delta_target: float = DELTA_TARGET,
        require_oi: bool = True,
        max_strike_dist: float = MAX_STRIKE_DIST,
    ) -> pd.Series | None:
        """
        Find the best strike within the given delta range.
        Gates: bid/ask always; OI only when require_oi=True.
        Picks the candidate closest to delta_target.

        Executability prefilter (vectorized, before the per-row delta loop):
          - strike within max_strike_dist of spot — far-OTM strikes never fill
          - bid >= MIN_LEG_BID — sub-$0.05 quotes are dead contracts
        """
        if chain_df is None or chain_df.empty:
            return None

        lo, hi = self.S * (1 - max_strike_dist), self.S * (1 + max_strike_dist)
        chain_df = chain_df[
            chain_df["strike"].between(lo, hi)
            & (chain_df["bid"].fillna(0) >= MIN_LEG_BID)
            & (chain_df["ask"].fillna(0) > 0)
        ]
        if chain_df.empty:
            return None

        candidates: list[tuple[float, pd.Series]] = []

        for _, row in chain_df.iterrows():
            K = float(row.get("strike", 0))
            if K <= 0:
                continue

            raw_iv = row.get("iv_pct", None)
            iv_pct = None if (raw_iv is None or pd.isna(raw_iv) or raw_iv <= 0) else float(raw_iv)
            dte = int(row.get("dte", 35))

            delta_val = get_delta(flag, self.S, K, iv_pct, dte, self.r)
            abs_delta = abs(delta_val)

            if not (delta_low <= abs_delta <= delta_high):
                continue
            if not passes_bidask_gate(row, self.ticker):
                continue
            if require_oi and not passes_oi_gate(row):
                continue

            candidates.append((abs(abs_delta - delta_target), row.copy()))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _find_long_leg(
        self,
        chain_df: pd.DataFrame,
        short_strike: float,
        flag: str,
        short_bid: float,
    ) -> pd.Series | None:
        """Find the narrowest long leg that passes bid/ask gate and yield-to-risk gate."""
        for width in SPREAD_WIDTH_RANGE:
            target_strike = short_strike + width if flag == "c" else short_strike - width

            if flag == "c":
                eligible = chain_df[chain_df["strike"] > short_strike]
            else:
                eligible = chain_df[chain_df["strike"] < short_strike]

            if eligible.empty:
                continue

            idx = (eligible["strike"] - target_strike).abs().idxmin()
            row = eligible.loc[idx]

            if not passes_bidask_gate(row, self.ticker):
                continue

            long_ask = float(row.get("ask", np.nan))
            if pd.isna(long_ask) or long_ask < 0:
                continue

            credit = short_bid - long_ask
            actual_width = abs(float(row["strike"]) - short_strike)
            if actual_width <= 0 or credit <= 0:
                continue

            if passes_yield_gate(credit, actual_width, self.min_credit_pct):
                return row

        return None

    def _compute_credit(self, short_leg: pd.Series, long_leg: pd.Series) -> float | None:
        short_bid = float(short_leg.get("bid", np.nan))
        long_ask = float(long_leg.get("ask", np.nan))
        if pd.isna(short_bid) or pd.isna(long_ask) or short_bid <= 0 or long_ask < 0:
            return None
        credit = short_bid - long_ask
        return credit if credit > 0 else None

    def _get_atm_iv(self) -> float | None:
        """Returns current ATM IV as a decimal (e.g. 0.35 for 35%)."""
        if not self.options_data:
            return None
        try:
            atm_ivs: list[float] = []
            for chain_df in (
                self.options_data.get("calls", pd.DataFrame()),
                self.options_data.get("puts", pd.DataFrame()),
            ):
                if chain_df.empty:
                    continue
                window = chain_df[(chain_df["dte"] >= 30) & (chain_df["dte"] <= 45)]
                if window.empty:
                    continue
                idx = (window["strike"] - self.S).abs().idxmin()
                iv_pct = window.loc[idx, "iv_pct"]
                if pd.notna(iv_pct) and float(iv_pct) > 0:
                    atm_ivs.append(float(iv_pct) / 100.0)
            return float(np.mean(atm_ivs)) if atm_ivs else None
        except Exception:
            return None

    def _build_result(
        self,
        strategy: str,
        short_leg: pd.Series,
        long_leg: pd.Series,
        credit: float,
        short_delta: float,
    ) -> dict:
        short_strike = float(short_leg["strike"])
        long_strike = float(long_leg["strike"])
        spread_width = abs(long_strike - short_strike)
        short_oi = int(short_leg.get("openInterest", 0))

        return {
            "ticker": self.ticker,
            "current_price": round(self.S, 2),
            "macro_trend": self.signals["macro_trend"],
            "rsi9": self.signals["rsi9"],
            "iv_rank": self.signals["iv_rank"],
            "strategy": strategy,
            "short_strike": short_strike,
            "long_strike": long_strike,
            "dte": int(short_leg.get("dte", 0)),
            "expiration": str(short_leg.get("expiration", "")),
            "short_bid": round(float(short_leg.get("bid", 0)), 2),
            "short_ask": round(float(short_leg.get("ask", 0)), 2),
            "long_bid": round(float(long_leg.get("bid", 0)), 2),
            "long_ask": round(float(long_leg.get("ask", 0)), 2),
            "credit": round(credit, 2),
            "spread_width": round(spread_width, 2),
            "credit_pct": round(credit / spread_width * 100, 1) if spread_width > 0 else 0.0,
            "max_profit": round(credit * 100, 2),
            "max_loss": round((spread_width - credit) * 100, 2),
            "short_delta": round(short_delta, 3),
            "short_oi": short_oi,
            "liquidity_warning": not passes_bidask_gate(short_leg, self.ticker),
        }

    def evaluate_call_credit_spread(self) -> dict | None:
        mt = self.signals["macro_trend"]
        rsi9 = self.signals["rsi9"]

        group_a = mt in ("BEARISH", "NEUTRAL")
        group_b = (
            mt == "BULLISH"
            and rsi9 > 72
            and self.signals["bb_upper_touch"]
            and self.signals["high_vol_env"]
        )

        if not (group_a or group_b):
            return None
        if not self.options_data:
            return None

        calls = self._select_expiration_window(
            self.options_data.get("calls", pd.DataFrame())
        )
        if calls.empty:
            return None

        short_leg = self._find_short_leg(calls, "c")
        if short_leg is None:
            return None

        short_strike = float(short_leg["strike"])
        short_bid = float(short_leg.get("bid", np.nan))
        if pd.isna(short_bid) or short_bid <= 0:
            return None

        long_leg = self._find_long_leg(calls, short_strike, "c", short_bid)
        if long_leg is None:
            return None

        credit = self._compute_credit(short_leg, long_leg)
        if credit is None:
            return None

        spread_width = abs(float(long_leg["strike"]) - short_strike)
        if not passes_yield_gate(credit, spread_width, self.min_credit_pct):
            return None

        iv_pct = short_leg.get("iv_pct", None)
        iv_pct_val = None if (iv_pct is None or pd.isna(iv_pct)) else float(iv_pct)
        dte = int(short_leg.get("dte", 35))
        delta_val = get_delta("c", self.S, short_strike, iv_pct_val, dte, self.r)

        logger.info(
            f"[{self.ticker}] CCS: short={short_strike} long={float(long_leg['strike'])} "
            f"credit=${credit:.2f} delta={delta_val:.3f}"
        )
        return self._build_result("CALL_CREDIT_SPREAD", short_leg, long_leg, credit, delta_val)

    def evaluate_put_credit_spread(self) -> dict | None:
        mt = self.signals["macro_trend"]

        if not (
            mt == "BULLISH"
            and self.signals["rsi9"] < 28
            and self.signals["bb_lower_touch"]
            and self.signals["high_vol_env"]
        ):
            return None
        if not self.options_data:
            return None

        puts = self._select_expiration_window(
            self.options_data.get("puts", pd.DataFrame())
        )
        if puts.empty:
            return None

        short_leg = self._find_short_leg(puts, "p")
        if short_leg is None:
            return None

        short_strike = float(short_leg["strike"])
        short_bid = float(short_leg.get("bid", np.nan))
        if pd.isna(short_bid) or short_bid <= 0:
            return None

        long_leg = self._find_long_leg(puts, short_strike, "p", short_bid)
        if long_leg is None:
            return None

        credit = self._compute_credit(short_leg, long_leg)
        if credit is None:
            return None

        spread_width = abs(short_strike - float(long_leg["strike"]))
        if not passes_yield_gate(credit, spread_width, self.min_credit_pct):
            return None

        iv_pct = short_leg.get("iv_pct", None)
        iv_pct_val = None if (iv_pct is None or pd.isna(iv_pct)) else float(iv_pct)
        dte = int(short_leg.get("dte", 35))
        delta_val = get_delta("p", self.S, short_strike, iv_pct_val, dte, self.r)

        logger.info(
            f"[{self.ticker}] PCS: short={short_strike} long={float(long_leg['strike'])} "
            f"credit=${credit:.2f} delta={delta_val:.3f}"
        )
        return self._build_result("PUT_CREDIT_SPREAD", short_leg, long_leg, credit, delta_val)

    def evaluate_iron_condor(
        self, ccs_result: dict | None, pcs_result: dict | None
    ) -> dict | None:
        if ccs_result is None or pcs_result is None:
            return None

        # Sanity: put short must be strictly below call short
        if pcs_result["short_strike"] >= ccs_result["short_strike"]:
            logger.warning(f"[{self.ticker}] IC rejected: inverted strikes")
            return None

        combined_credit = ccs_result["credit"] + pcs_result["credit"]
        wider_width = max(ccs_result["spread_width"], pcs_result["spread_width"])

        if not passes_yield_gate(combined_credit, wider_width, self.min_credit_pct):
            return None

        return {
            "ticker": self.ticker,
            "current_price": ccs_result["current_price"],
            "macro_trend": self.signals["macro_trend"],
            "rsi9": self.signals["rsi9"],
            "iv_rank": self.signals["iv_rank"],
            "strategy": "IRON_CONDOR",
            # short_strike = put short (lower boundary), long_strike = call short (upper)
            "short_strike": pcs_result["short_strike"],
            "long_strike": ccs_result["short_strike"],
            "dte": pcs_result["dte"],
            "expiration": pcs_result["expiration"],
            "short_bid": pcs_result["short_bid"],
            "short_ask": pcs_result["short_ask"],
            "long_bid": ccs_result["short_bid"],
            "long_ask": ccs_result["short_ask"],
            "credit": round(combined_credit, 2),
            "spread_width": round(wider_width, 2),
            "credit_pct": round(combined_credit / wider_width * 100, 1),
            "max_profit": round(combined_credit * 100, 2),
            "max_loss": round((wider_width - combined_credit) * 100, 2),
            "short_delta": pcs_result["short_delta"],
            "short_oi": min(pcs_result["short_oi"], ccs_result["short_oi"]),
            "liquidity_warning": pcs_result["liquidity_warning"] or ccs_result["liquidity_warning"],
            # IC-specific leg detail (used in signal detail expander)
            "put_short_strike": pcs_result["short_strike"],
            "put_long_strike": pcs_result["long_strike"],
            "call_short_strike": ccs_result["short_strike"],
            "call_long_strike": ccs_result["long_strike"],
        }

    # ── Call Calendar Spread ──────────────────────────────────────────────────

    def _find_calendar_legs(
        self, calls_df: pd.DataFrame
    ) -> tuple[pd.Series | None, pd.Series | None]:
        """
        Front month (short): 15–30 DTE standard monthly.
        Back month (long):   45–65 DTE standard monthly, same strike.
        Strike: closest to ATM on the front month.
        """
        front = calls_df[(calls_df["dte"] >= 15) & (calls_df["dte"] <= 30)]
        back = calls_df[(calls_df["dte"] >= 45) & (calls_df["dte"] <= 65)]

        if front.empty or back.empty:
            return None, None

        # ATM strike in front month
        front_atm_idx = (front["strike"] - self.S).abs().idxmin()
        front_leg = front.loc[front_atm_idx]

        if not passes_bidask_gate(front_leg, self.ticker):
            return None, None
        if not passes_oi_gate(front_leg):
            return None, None

        # Same (or closest available) strike in back month
        target_strike = float(front_leg["strike"])
        exact = back[abs(back["strike"] - target_strike) < 0.01]
        if not exact.empty:
            back_leg = exact.iloc[0]
        else:
            back_leg = back.loc[(back["strike"] - target_strike).abs().idxmin()]

        if not passes_bidask_gate(back_leg, self.ticker):
            return None, None

        return front_leg, back_leg

    def _base_fields(self, strategy: str) -> dict:
        """Common result-dict fields shared by every strategy."""
        return {
            "ticker": self.ticker,
            "current_price": round(self.S, 2),
            "macro_trend": self.signals["macro_trend"],
            "rsi9": round(self.signals["rsi9"], 2),
            "iv_rank": round(self.signals["iv_rank"], 2),
            "strategy": strategy,
        }

    def _leg_delta(self, leg: pd.Series, flag: str) -> float:
        raw_iv = leg.get("iv_pct", None)
        iv = None if (raw_iv is None or pd.isna(raw_iv) or raw_iv <= 0) else float(raw_iv)
        return get_delta(flag, self.S, float(leg["strike"]), iv, int(leg.get("dte", 35)), self.r)

    def _paired_window(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Calls and puts filtered to ONE common expiration nearest 37 DTE.
        Used by multi-leg strategies (straddle, strangle, butterfly) where all
        legs must share the same expiry."""
        if not self.options_data:
            return pd.DataFrame(), pd.DataFrame()
        calls = self._select_expiration_window(self.options_data.get("calls", pd.DataFrame()))
        puts = self._select_expiration_window(self.options_data.get("puts", pd.DataFrame()))
        if calls.empty or puts.empty:
            return pd.DataFrame(), pd.DataFrame()
        common = set(calls["expiration"]) & set(puts["expiration"])
        if not common:
            return pd.DataFrame(), pd.DataFrame()
        dte_map = calls[calls["expiration"].isin(common)].groupby("expiration")["dte"].first()
        best = (dte_map - 37).abs().idxmin()
        return (
            calls[calls["expiration"] == best].reset_index(drop=True),
            puts[puts["expiration"] == best].reset_index(drop=True),
        )

    def _evaluate_calendar(self, flag: str) -> dict | None:
        """
        Calendar spread: sell the front month (15–30 DTE), buy the back month
        (45–65 DTE) at the same near-ATM strike, for a net debit. Profits when
        price pins the strike and front-month theta outpaces the back month.
        flag: "c" → CALL_CALENDAR_SPREAD · "p" → PUT_CALENDAR_SPREAD
        """
        key = "CALL_CALENDAR_SPREAD" if flag == "c" else "PUT_CALENDAR_SPREAD"
        if not STRATEGY_TRIGGERS[key](self.signals):
            return None
        if not self.options_data:
            return None

        chain = self.options_data.get("calls" if flag == "c" else "puts", pd.DataFrame())
        if chain.empty:
            return None

        front_leg, back_leg = self._find_calendar_legs(chain)
        if front_leg is None or back_leg is None:
            return None

        long_ask = float(back_leg.get("ask", np.nan))
        short_bid = float(front_leg.get("bid", np.nan))
        if pd.isna(long_ask) or pd.isna(short_bid) or long_ask <= 0 or short_bid < 0:
            return None

        net_debit = long_ask - short_bid
        if net_debit <= 0:
            return None
        # Reject if debit exceeds 66% of the back month's ask (poor risk/reward)
        if net_debit > 0.66 * long_ask:
            return None

        short_strike = float(front_leg["strike"])
        front_dte = int(front_leg.get("dte", 20))
        back_dte = int(back_leg.get("dte", 50))
        delta_val = self._leg_delta(front_leg, flag)

        logger.info(
            f"[{self.ticker}] {key}: strike={short_strike} "
            f"front_dte={front_dte} back_dte={back_dte} "
            f"debit=${net_debit:.2f} delta={delta_val:.3f}"
        )

        return {
            **self._base_fields(key),
            "short_strike": short_strike,
            "long_strike": float(back_leg["strike"]),  # same/nearest strike, back month
            "dte": front_dte,                          # front month DTE shown in main table
            "expiration": str(front_leg.get("expiration", "")),
            "short_bid": round(short_bid, 2),
            "short_ask": round(float(front_leg.get("ask", 0)), 2),
            "long_bid": round(float(back_leg.get("bid", 0)), 2),
            "long_ask": round(long_ask, 2),
            "net_debit": round(net_debit, 2),
            "credit": None,
            "spread_width": 0.0,
            "credit_pct": 0.0,
            "max_profit": round(float(back_leg.get("bid", 0)) * 100, 2),
            "max_loss": round(net_debit * 100, 2),
            "short_delta": round(delta_val, 3),
            "short_oi": int(front_leg.get("openInterest", 0)),
            "liquidity_warning": not passes_bidask_gate(front_leg, self.ticker),
            # Calendar-specific detail (shown in signal expander)
            "front_expiration": str(front_leg.get("expiration", "")),
            "front_dte": front_dte,
            "back_expiration": str(back_leg.get("expiration", "")),
            "back_dte": back_dte,
        }

    def evaluate_call_calendar_spread(self) -> dict | None:
        return self._evaluate_calendar("c")

    def evaluate_put_calendar_spread(self) -> dict | None:
        return self._evaluate_calendar("p")

    # ── Diagonal spreads ──────────────────────────────────────────────────────

    def _evaluate_diagonal(self, flag: str) -> dict | None:
        """
        Diagonal spread — a calendar + vertical hybrid:
          LONG  back-month ITM option (delta ~0.70, 45–65 DTE) — the stock substitute
          SHORT front-month OTM option (delta ~0.25, 15–30 DTE) — the income leg
        The call version is the classic "Poor Man's Covered Call" (PMCC).
        Assignment-safety gate: net debit < strike width, so the position still
        closes profitably even if the short strike is breached at expiry.
        flag: "c" → CALL_DIAGONAL_SPREAD · "p" → PUT_DIAGONAL_SPREAD
        """
        key = "CALL_DIAGONAL_SPREAD" if flag == "c" else "PUT_DIAGONAL_SPREAD"
        if not STRATEGY_TRIGGERS[key](self.signals):
            return None
        if not self.options_data:
            return None

        chain = self.options_data.get("calls" if flag == "c" else "puts", pd.DataFrame())
        if chain.empty:
            return None

        front = chain[(chain["dte"] >= 15) & (chain["dte"] <= 30)]
        back = chain[(chain["dte"] >= 45) & (chain["dte"] <= 65)]
        if front.empty or back.empty:
            return None

        # Long leg: back-month ITM (delta 0.60–0.80). OI gate required —
        # this is the capital-intensive leg you may need to exit early.
        long_leg = self._find_short_leg(
            back, flag, delta_low=0.60, delta_high=0.80, delta_target=0.70,
        )
        if long_leg is None:
            return None
        long_K = float(long_leg["strike"])
        long_ask = float(long_leg.get("ask", np.nan))
        if pd.isna(long_ask) or long_ask <= 0:
            return None

        # Short leg: front-month OTM beyond both spot and the long strike
        if flag == "c":
            otm = front[front["strike"] > max(long_K, self.S)]
        else:
            otm = front[front["strike"] < min(long_K, self.S)]
        if otm.empty:
            return None
        short_leg = self._find_short_leg(
            otm, flag, delta_low=0.15, delta_high=0.30, delta_target=0.25,
            require_oi=False,
        )
        if short_leg is None:
            return None
        short_K = float(short_leg["strike"])
        short_bid = float(short_leg.get("bid", np.nan))
        if pd.isna(short_bid) or short_bid <= 0:
            return None

        net_debit = long_ask - short_bid
        if net_debit <= 0:
            return None
        width = abs(short_K - long_K)
        if width <= 0 or net_debit >= width:  # assignment-safety gate
            return None

        front_dte = int(short_leg.get("dte", 20))
        back_dte = int(long_leg.get("dte", 50))
        delta_val = self._leg_delta(long_leg, flag)

        logger.info(
            f"[{self.ticker}] {key}: long={long_K} ({back_dte}d) short={short_K} "
            f"({front_dte}d) debit=${net_debit:.2f} width={width:.1f}"
        )
        return {
            **self._base_fields(key),
            "long_strike": long_K,      # back-month stock substitute
            "short_strike": short_K,    # front-month income leg
            "dte": front_dte,           # short leg DTE shown in main table
            "expiration": str(short_leg.get("expiration", "")),
            "long_bid": round(float(long_leg.get("bid", 0)), 2),
            "long_ask": round(long_ask, 2),
            "short_bid": round(short_bid, 2),
            "short_ask": round(float(short_leg.get("ask", 0)), 2),
            "net_debit": round(net_debit, 2),
            "credit": None,
            "spread_width": round(width, 2),
            "credit_pct": 0.0,
            "debit_pct": round(net_debit / width * 100, 1),
            "max_profit": round((width - net_debit) * 100, 2),  # if assigned at short strike
            "max_loss": round(net_debit * 100, 2),
            "short_delta": round(delta_val, 3),
            "short_oi": int(long_leg.get("openInterest", 0)),
            "liquidity_warning": False,
            "front_expiration": str(short_leg.get("expiration", "")),
            "front_dte": front_dte,
            "back_expiration": str(long_leg.get("expiration", "")),
            "back_dte": back_dte,
        }

    def evaluate_call_diagonal_spread(self) -> dict | None:
        return self._evaluate_diagonal("c")

    def evaluate_put_diagonal_spread(self) -> dict | None:
        return self._evaluate_diagonal("p")

    # ── Bull Call Debit Spread ────────────────────────────────────────────────

    def evaluate_bull_call_debit_spread(self) -> dict | None:
        """
        Bull Call Debit Spread — buy a near-ATM call, sell an OTM call (same expiry).
        Net DEBIT paid upfront. Profit if stock rises above the long strike at expiry.
        Max loss = debit paid. Max profit = (spread width − debit) × 100 per contract.

        Triggers (all required):
          - BULLISH trend (EMA-50 above EMA-200)
          - RSI(9) in [50, 75]: upward momentum, not at overbought extreme
          - IV Rank < 50: lower IV → cheaper premiums → smaller debit to pay
        """
        if self.signals["macro_trend"] != "BULLISH":
            return None
        rsi9 = self.signals["rsi9"]
        if not (50 <= rsi9 <= 75):
            return None
        if self.signals["iv_rank"] >= 50:
            return None
        if not self.options_data:
            return None

        calls = self._select_expiration_window(
            self.options_data.get("calls", pd.DataFrame())
        )
        if calls.empty:
            return None

        # Long leg: near-ATM call (abs delta 0.35–0.65, target 0.50). OI gate required.
        long_leg = self._find_short_leg(
            calls, "c",
            delta_low=0.35, delta_high=0.65, delta_target=0.50,
            require_oi=True,
        )
        if long_leg is None:
            return None

        long_strike = float(long_leg["strike"])
        long_ask = float(long_leg.get("ask", np.nan))
        if pd.isna(long_ask) or long_ask <= 0:
            return None

        # Short leg: OTM call above the long strike (abs delta 0.15–0.35). No OI gate.
        otm_calls = calls[calls["strike"] > long_strike]
        if otm_calls.empty:
            return None
        short_leg = self._find_short_leg(
            otm_calls, "c",
            delta_low=0.15, delta_high=0.35, delta_target=0.25,
            require_oi=False,
        )
        if short_leg is None:
            return None

        short_strike = float(short_leg["strike"])
        short_bid = float(short_leg.get("bid", np.nan))
        if pd.isna(short_bid) or short_bid < 0:
            return None

        net_debit = long_ask - short_bid
        if net_debit <= 0:
            return None

        spread_width = short_strike - long_strike
        if spread_width <= 0:
            return None

        # Gate: debit must be < 75% of width → max profit ≥ 25% of width
        if net_debit >= 0.75 * spread_width:
            return None

        iv_pct = long_leg.get("iv_pct", None)
        iv_pct_val = None if (iv_pct is None or pd.isna(iv_pct)) else float(iv_pct)
        dte = int(long_leg.get("dte", 35))
        delta_val = get_delta("c", self.S, long_strike, iv_pct_val, dte, self.r)

        logger.info(
            f"[{self.ticker}] BCDS: long={long_strike} short={short_strike} "
            f"debit=${net_debit:.2f} width={spread_width:.1f} delta={delta_val:.3f}"
        )
        return {
            "ticker": self.ticker,
            "current_price": round(self.S, 2),
            "macro_trend": self.signals["macro_trend"],
            "rsi9": round(rsi9, 2),
            "iv_rank": round(self.signals["iv_rank"], 2),
            "strategy": "BULL_CALL_DEBIT_SPREAD",
            "long_strike": long_strike,
            "short_strike": short_strike,
            "dte": dte,
            "expiration": str(long_leg.get("expiration", "")),
            "long_bid": round(float(long_leg.get("bid", 0)), 2),
            "long_ask": round(long_ask, 2),
            "short_bid": round(short_bid, 2),
            "short_ask": round(float(short_leg.get("ask", 0)), 2),
            "net_debit": round(net_debit, 2),
            "credit": None,
            "spread_width": round(spread_width, 2),
            "credit_pct": 0.0,
            "debit_pct": round(net_debit / spread_width * 100, 1),
            "max_profit": round((spread_width - net_debit) * 100, 2),
            "max_loss": round(net_debit * 100, 2),
            "short_delta": round(delta_val, 3),
            "short_oi": int(long_leg.get("openInterest", 0)),  # primary leg needs liquidity
            "liquidity_warning": not passes_bidask_gate(long_leg, self.ticker),
        }

    # ── Bear Put Debit Spread ─────────────────────────────────────────────────

    def evaluate_bear_put_debit_spread(self) -> dict | None:
        """
        Bear Put Debit Spread — buy a near-ATM put, sell an OTM put (same expiry).
        Net DEBIT paid upfront. Profit if stock falls below the long strike at expiry.
        Max loss = debit paid. Max profit = (spread width − debit) × 100 per contract.

        Triggers (all required):
          - BEARISH or NEUTRAL trend (EMA-50 at or below EMA-200)
          - RSI(9) in [25, 50]: downward momentum
          - IV Rank < 50: lower IV → cheaper premiums → smaller debit to pay
        """
        mt = self.signals["macro_trend"]
        if mt not in ("BEARISH", "NEUTRAL"):
            return None
        rsi9 = self.signals["rsi9"]
        if not (25 <= rsi9 <= 50):
            return None
        if self.signals["iv_rank"] >= 50:
            return None
        if not self.options_data:
            return None

        puts = self._select_expiration_window(
            self.options_data.get("puts", pd.DataFrame())
        )
        if puts.empty:
            return None

        # Long leg: near-ATM put (abs delta 0.35–0.65, target 0.50). OI gate required.
        long_leg = self._find_short_leg(
            puts, "p",
            delta_low=0.35, delta_high=0.65, delta_target=0.50,
            require_oi=True,
        )
        if long_leg is None:
            return None

        long_strike = float(long_leg["strike"])
        long_ask = float(long_leg.get("ask", np.nan))
        if pd.isna(long_ask) or long_ask <= 0:
            return None

        # Short leg: OTM put below the long strike (abs delta 0.15–0.35). No OI gate.
        otm_puts = puts[puts["strike"] < long_strike]
        if otm_puts.empty:
            return None
        short_leg = self._find_short_leg(
            otm_puts, "p",
            delta_low=0.15, delta_high=0.35, delta_target=0.25,
            require_oi=False,
        )
        if short_leg is None:
            return None

        short_strike = float(short_leg["strike"])
        short_bid = float(short_leg.get("bid", np.nan))
        if pd.isna(short_bid) or short_bid < 0:
            return None

        net_debit = long_ask - short_bid
        if net_debit <= 0:
            return None

        spread_width = long_strike - short_strike
        if spread_width <= 0:
            return None

        # Gate: debit must be < 75% of width → max profit ≥ 25% of width
        if net_debit >= 0.75 * spread_width:
            return None

        iv_pct = long_leg.get("iv_pct", None)
        iv_pct_val = None if (iv_pct is None or pd.isna(iv_pct)) else float(iv_pct)
        dte = int(long_leg.get("dte", 35))
        delta_val = get_delta("p", self.S, long_strike, iv_pct_val, dte, self.r)

        logger.info(
            f"[{self.ticker}] BPDS: long={long_strike} short={short_strike} "
            f"debit=${net_debit:.2f} width={spread_width:.1f} delta={delta_val:.3f}"
        )
        return {
            "ticker": self.ticker,
            "current_price": round(self.S, 2),
            "macro_trend": mt,
            "rsi9": round(rsi9, 2),
            "iv_rank": round(self.signals["iv_rank"], 2),
            "strategy": "BEAR_PUT_DEBIT_SPREAD",
            "long_strike": long_strike,
            "short_strike": short_strike,
            "dte": dte,
            "expiration": str(long_leg.get("expiration", "")),
            "long_bid": round(float(long_leg.get("bid", 0)), 2),
            "long_ask": round(long_ask, 2),
            "short_bid": round(short_bid, 2),
            "short_ask": round(float(short_leg.get("ask", 0)), 2),
            "net_debit": round(net_debit, 2),
            "credit": None,
            "spread_width": round(spread_width, 2),
            "credit_pct": 0.0,
            "debit_pct": round(net_debit / spread_width * 100, 1),
            "max_profit": round((spread_width - net_debit) * 100, 2),
            "max_loss": round(net_debit * 100, 2),
            "short_delta": round(delta_val, 3),
            "short_oi": int(long_leg.get("openInterest", 0)),  # primary leg needs liquidity
            "liquidity_warning": not passes_bidask_gate(long_leg, self.ticker),
        }

    # ── Premium income: Cash-Secured Put / Covered Call ──────────────────────

    def evaluate_cash_secured_put(self) -> dict | None:
        """
        Sell an OTM put backed by cash collateral (strike × 100).
        Profit: keep full premium if stock stays above the strike; otherwise
        you're assigned shares at an effective discount.
        Trigger: bullish/neutral trend + IV Rank > 45 (rich premiums).
        """
        if not STRATEGY_TRIGGERS["CASH_SECURED_PUT"](self.signals):
            return None
        if not self.options_data:
            return None
        puts = self._select_expiration_window(self.options_data.get("puts", pd.DataFrame()))
        if puts.empty:
            return None

        leg = self._find_short_leg(puts, "p", delta_low=0.15, delta_high=0.30, delta_target=0.22)
        if leg is None:
            return None
        K = float(leg["strike"])
        bid = float(leg.get("bid", np.nan))
        if pd.isna(bid) or bid <= 0 or K >= self.S:  # must be OTM (below spot)
            return None

        delta_val = self._leg_delta(leg, "p")
        logger.info(f"[{self.ticker}] CSP: strike={K} credit=${bid:.2f} delta={delta_val:.3f}")
        return {
            **self._base_fields("CASH_SECURED_PUT"),
            "short_strike": K,
            "long_strike": None,
            "dte": int(leg.get("dte", 0)),
            "expiration": str(leg.get("expiration", "")),
            "short_bid": round(bid, 2),
            "short_ask": round(float(leg.get("ask", 0)), 2),
            "credit": round(bid, 2),
            "spread_width": 0.0,
            "credit_pct": round(bid / K * 100, 1),       # yield on cash collateral
            "max_profit": round(bid * 100, 2),
            "max_loss": round((K - bid) * 100, 2),        # stock to zero, less premium
            "short_delta": round(delta_val, 3),
            "short_oi": int(leg.get("openInterest", 0)),
            "liquidity_warning": False,
        }

    def evaluate_covered_call(self) -> dict | None:
        """
        Sell an OTM call against 100 owned shares (requires share ownership).
        Profit: keep premium if stock stays below the strike; shares called
        away at the strike (still profitable) if it rallies through.
        Trigger: bullish/neutral trend + IV Rank > 40, not overbought.
        """
        if not STRATEGY_TRIGGERS["COVERED_CALL"](self.signals):
            return None
        if not self.options_data:
            return None
        calls = self._select_expiration_window(self.options_data.get("calls", pd.DataFrame()))
        if calls.empty:
            return None

        leg = self._find_short_leg(calls, "c", delta_low=0.20, delta_high=0.35, delta_target=0.28)
        if leg is None:
            return None
        K = float(leg["strike"])
        bid = float(leg.get("bid", np.nan))
        if pd.isna(bid) or bid <= 0 or K <= self.S:  # must be OTM (above spot)
            return None

        delta_val = self._leg_delta(leg, "c")
        logger.info(f"[{self.ticker}] CC: strike={K} credit=${bid:.2f} delta={delta_val:.3f}")
        return {
            **self._base_fields("COVERED_CALL"),
            "short_strike": K,
            "long_strike": None,
            "dte": int(leg.get("dte", 0)),
            "expiration": str(leg.get("expiration", "")),
            "short_bid": round(bid, 2),
            "short_ask": round(float(leg.get("ask", 0)), 2),
            "credit": round(bid, 2),
            "spread_width": 0.0,
            "credit_pct": round(bid / self.S * 100, 1),    # yield on share cost
            "max_profit": round((K - self.S + bid) * 100, 2),
            "max_loss": round((self.S - bid) * 100, 2),     # shares to zero, less premium
            "short_delta": round(delta_val, 3),
            "short_oi": int(leg.get("openInterest", 0)),
            "liquidity_warning": False,
        }

    # ── Iron Butterfly ────────────────────────────────────────────────────────

    def evaluate_iron_butterfly(self) -> dict | None:
        """
        Sell ATM call + ATM put at the same strike, buy protective wings.
        Maximum theta harvest; profits if price pins the body at expiry.
        Trigger: neutral RSI (40–60) + high IV environment, no BB touch.
        """
        if not STRATEGY_TRIGGERS["IRON_BUTTERFLY"](self.signals):
            return None
        calls, puts = self._paired_window()
        if calls.empty or puts.empty:
            return None

        strikes = sorted(set(calls["strike"]) & set(puts["strike"]))
        if not strikes:
            return None
        atm_K = min(strikes, key=lambda k: abs(k - self.S))
        if abs(atm_K - self.S) / self.S > 0.05:  # body must hug spot
            return None

        atm_call = calls[calls["strike"] == atm_K].iloc[0]
        atm_put = puts[puts["strike"] == atm_K].iloc[0]
        for leg in (atm_call, atm_put):
            if not passes_bidask_gate(leg, self.ticker) or not passes_oi_gate(leg):
                return None

        # Wings 3–6% of spot away — nearest available strikes that pass gates
        for wing_pct in (0.03, 0.04, 0.05, 0.06):
            w = self.S * wing_pct
            up = calls[calls["strike"] >= atm_K + w]
            dn = puts[puts["strike"] <= atm_K - w]
            if up.empty or dn.empty:
                continue
            cw = up.loc[up["strike"].idxmin()]
            pw = dn.loc[dn["strike"].idxmax()]
            if not passes_bidask_gate(cw, self.ticker) or not passes_bidask_gate(pw, self.ticker):
                continue

            credit = (float(atm_call["bid"]) + float(atm_put["bid"])
                      - float(cw["ask"]) - float(pw["ask"]))
            width = max(float(cw["strike"]) - atm_K, atm_K - float(pw["strike"]))
            if credit <= 0 or width <= 0:
                continue
            if not passes_yield_gate(credit, width, self.min_credit_pct):
                continue

            delta_val = self._leg_delta(atm_call, "c")
            logger.info(
                f"[{self.ticker}] IRON_BUTTERFLY: body={atm_K} wings="
                f"{float(pw['strike'])}/{float(cw['strike'])} credit=${credit:.2f}"
            )
            return {
                **self._base_fields("IRON_BUTTERFLY"),
                "short_strike": atm_K,                 # body (both short legs)
                "long_strike": float(cw["strike"]),    # call wing (put wing in detail)
                "dte": int(atm_call.get("dte", 0)),
                "expiration": str(atm_call.get("expiration", "")),
                "short_bid": round(float(atm_call["bid"]) + float(atm_put["bid"]), 2),
                "short_ask": round(float(atm_call.get("ask", 0)) + float(atm_put.get("ask", 0)), 2),
                "long_bid": round(float(cw.get("bid", 0)) + float(pw.get("bid", 0)), 2),
                "long_ask": round(float(cw["ask"]) + float(pw["ask"]), 2),
                "credit": round(credit, 2),
                "spread_width": round(width, 2),
                "credit_pct": round(credit / width * 100, 1),
                "max_profit": round(credit * 100, 2),
                "max_loss": round((width - credit) * 100, 2),
                "short_delta": round(delta_val, 3),
                "short_oi": min(int(atm_call.get("openInterest", 0)),
                                int(atm_put.get("openInterest", 0))),
                "liquidity_warning": False,
                "call_wing_strike": float(cw["strike"]),
                "put_wing_strike": float(pw["strike"]),
            }
        return None

    # ── Long volatility: Straddle / Strangle ──────────────────────────────────

    def evaluate_long_straddle(self) -> dict | None:
        """
        Buy ATM call + ATM put, same strike & expiry. Profits on a large move
        in EITHER direction (or an IV expansion).
        Trigger: IV Rank < 25 (cheap options) + neutral RSI + price coiled.
        Sanity gate: combined debit ≤ 12% of spot, else breakeven is unrealistic.
        """
        if not STRATEGY_TRIGGERS["LONG_STRADDLE"](self.signals):
            return None
        calls, puts = self._paired_window()
        if calls.empty or puts.empty:
            return None

        strikes = sorted(set(calls["strike"]) & set(puts["strike"]))
        if not strikes:
            return None
        atm_K = min(strikes, key=lambda k: abs(k - self.S))
        if abs(atm_K - self.S) / self.S > 0.05:
            return None

        c = calls[calls["strike"] == atm_K].iloc[0]
        p = puts[puts["strike"] == atm_K].iloc[0]
        for leg in (c, p):
            if not passes_bidask_gate(leg, self.ticker) or not passes_oi_gate(leg):
                return None

        debit = float(c.get("ask", np.nan)) + float(p.get("ask", np.nan))
        if pd.isna(debit) or debit <= 0:
            return None
        if debit / self.S > 0.12:  # breakeven move > 12% — too expensive
            return None

        logger.info(f"[{self.ticker}] LONG_STRADDLE: strike={atm_K} debit=${debit:.2f}")
        return {
            **self._base_fields("LONG_STRADDLE"),
            "short_strike": None,
            "long_strike": atm_K,
            "dte": int(c.get("dte", 0)),
            "expiration": str(c.get("expiration", "")),
            "long_bid": round(float(c.get("bid", 0)) + float(p.get("bid", 0)), 2),
            "long_ask": round(debit, 2),
            "net_debit": round(debit, 2),
            "credit": None,
            "spread_width": 0.0,
            "credit_pct": 0.0,
            "debit_pct": round(debit / self.S * 100, 1),   # breakeven move as % of spot
            "max_profit": float("nan"),                     # unlimited (upside leg)
            "max_loss": round(debit * 100, 2),
            "short_delta": round(self._leg_delta(c, "c"), 3),
            "short_oi": min(int(c.get("openInterest", 0)), int(p.get("openInterest", 0))),
            "liquidity_warning": False,
        }

    def evaluate_long_strangle(self) -> dict | None:
        """
        Buy OTM call + OTM put (delta ≈ 0.25 each), same expiry.
        Cheaper than a straddle; needs a bigger move to pay off.
        Trigger: IV Rank < 25 + neutral RSI + price coiled.
        Sanity gate: combined debit ≤ 8% of spot.
        """
        if not STRATEGY_TRIGGERS["LONG_STRANGLE"](self.signals):
            return None
        calls, puts = self._paired_window()
        if calls.empty or puts.empty:
            return None

        c_leg = self._find_short_leg(calls, "c", delta_low=0.20, delta_high=0.30, delta_target=0.25)
        p_leg = self._find_short_leg(puts, "p", delta_low=0.20, delta_high=0.30, delta_target=0.25)
        if c_leg is None or p_leg is None:
            return None

        debit = float(c_leg.get("ask", np.nan)) + float(p_leg.get("ask", np.nan))
        if pd.isna(debit) or debit <= 0:
            return None
        if debit / self.S > 0.08:
            return None

        call_K, put_K = float(c_leg["strike"]), float(p_leg["strike"])
        if put_K >= call_K:
            return None

        logger.info(f"[{self.ticker}] LONG_STRANGLE: {put_K}P/{call_K}C debit=${debit:.2f}")
        return {
            **self._base_fields("LONG_STRANGLE"),
            "short_strike": put_K,    # lower bound of the strangle (both legs are long)
            "long_strike": call_K,    # upper bound
            "dte": int(c_leg.get("dte", 0)),
            "expiration": str(c_leg.get("expiration", "")),
            "long_bid": round(float(c_leg.get("bid", 0)) + float(p_leg.get("bid", 0)), 2),
            "long_ask": round(debit, 2),
            "net_debit": round(debit, 2),
            "credit": None,
            "spread_width": 0.0,
            "credit_pct": 0.0,
            "debit_pct": round(debit / self.S * 100, 1),
            "max_profit": float("nan"),
            "max_loss": round(debit * 100, 2),
            "short_delta": round(self._leg_delta(c_leg, "c"), 3),
            "short_oi": min(int(c_leg.get("openInterest", 0)), int(p_leg.get("openInterest", 0))),
            "liquidity_warning": False,
        }

    def evaluate_short_strangle(self) -> dict | None:
        """
        Sell OTM call + OTM put (delta ≈ 0.15–0.20 each), same expiry.
        ⚠️ UNDEFINED RISK — losses are unlimited beyond either strike.
        Requires margin approval; shown for completeness with a risk warning.
        Trigger: neutral RSI (40–60) + high IV environment, no BB touch.
        """
        if not STRATEGY_TRIGGERS["SHORT_STRANGLE"](self.signals):
            return None
        calls, puts = self._paired_window()
        if calls.empty or puts.empty:
            return None

        c_leg = self._find_short_leg(calls, "c")
        p_leg = self._find_short_leg(puts, "p")
        if c_leg is None or p_leg is None:
            return None

        c_bid = float(c_leg.get("bid", np.nan))
        p_bid = float(p_leg.get("bid", np.nan))
        if pd.isna(c_bid) or pd.isna(p_bid) or c_bid <= 0 or p_bid <= 0:
            return None
        credit = c_bid + p_bid

        call_K, put_K = float(c_leg["strike"]), float(p_leg["strike"])
        if put_K >= call_K:
            return None

        logger.info(f"[{self.ticker}] SHORT_STRANGLE: {put_K}P/{call_K}C credit=${credit:.2f}")
        return {
            **self._base_fields("SHORT_STRANGLE"),
            "short_strike": put_K,    # lower short strike
            "long_strike": call_K,    # upper short strike (both legs are short)
            "dte": int(c_leg.get("dte", 0)),
            "expiration": str(c_leg.get("expiration", "")),
            "short_bid": round(credit, 2),
            "short_ask": round(float(c_leg.get("ask", 0)) + float(p_leg.get("ask", 0)), 2),
            "credit": round(credit, 2),
            "spread_width": 0.0,
            "credit_pct": 0.0,
            "max_profit": round(credit * 100, 2),
            "max_loss": float("nan"),  # UNDEFINED — naked short legs
            "short_delta": round(self._leg_delta(c_leg, "c"), 3),
            "short_oi": min(int(c_leg.get("openInterest", 0)), int(p_leg.get("openInterest", 0))),
            "liquidity_warning": True,  # flag undefined risk in the table
        }

    # ── Single-leg directional: Long Call / Long Put ──────────────────────────

    def _evaluate_long_single(self, flag: str) -> dict | None:
        """
        Buy a slightly-ITM option (delta 0.55–0.75): keeps intrinsic value and
        suffers less theta bleed than ATM/OTM lottery tickets.
        flag: "c" → LONG_CALL · "p" → LONG_PUT
        """
        key = "LONG_CALL" if flag == "c" else "LONG_PUT"
        if not STRATEGY_TRIGGERS[key](self.signals):
            return None
        if not self.options_data:
            return None
        chain = self._select_expiration_window(
            self.options_data.get("calls" if flag == "c" else "puts", pd.DataFrame())
        )
        if chain.empty:
            return None

        leg = self._find_short_leg(chain, flag, delta_low=0.55, delta_high=0.75, delta_target=0.65)
        if leg is None:
            return None
        K = float(leg["strike"])
        ask = float(leg.get("ask", np.nan))
        if pd.isna(ask) or ask <= 0:
            return None

        delta_val = self._leg_delta(leg, flag)
        breakeven = K + ask if flag == "c" else K - ask
        logger.info(f"[{self.ticker}] {key}: strike={K} debit=${ask:.2f} delta={delta_val:.3f}")
        return {
            **self._base_fields(key),
            "short_strike": None,
            "long_strike": K,
            "dte": int(leg.get("dte", 0)),
            "expiration": str(leg.get("expiration", "")),
            "long_bid": round(float(leg.get("bid", 0)), 2),
            "long_ask": round(ask, 2),
            "net_debit": round(ask, 2),
            "credit": None,
            "spread_width": 0.0,
            "credit_pct": 0.0,
            "debit_pct": round(ask / self.S * 100, 1),
            "max_profit": float("nan") if flag == "c" else round((K - ask) * 100, 2),
            "max_loss": round(ask * 100, 2),
            "short_delta": round(delta_val, 3),
            "short_oi": int(leg.get("openInterest", 0)),
            "liquidity_warning": False,
            "breakeven": round(breakeven, 2),
        }

    def evaluate_long_call(self) -> dict | None:
        return self._evaluate_long_single("c")

    def evaluate_long_put(self) -> dict | None:
        return self._evaluate_long_single("p")

    def run(self) -> list[dict]:
        ccs = self.evaluate_call_credit_spread()
        pcs = self.evaluate_put_credit_spread()
        results = [
            ccs,
            pcs,
            self.evaluate_iron_condor(ccs, pcs),
            self.evaluate_iron_butterfly(),
            self.evaluate_cash_secured_put(),
            self.evaluate_covered_call(),
            self.evaluate_short_strangle(),
            self.evaluate_call_diagonal_spread(),
            self.evaluate_put_diagonal_spread(),
            self.evaluate_call_calendar_spread(),
            self.evaluate_put_calendar_spread(),
            self.evaluate_bull_call_debit_spread(),
            self.evaluate_bear_put_debit_spread(),
            self.evaluate_long_call(),
            self.evaluate_long_put(),
            self.evaluate_long_straddle(),
            self.evaluate_long_strangle(),
        ]
        return [r for r in results if r is not None]


# ── Module-level entry point ──────────────────────────────────────────────────

def screen_triggered_tickers(
    phase1_data: dict,
    phase2_options: dict,
    signals_data: dict,
    r: float = 0.05,
    min_credit_pct: float = MIN_CREDIT_PCT,
) -> pd.DataFrame:
    results: list[dict] = []

    for ticker in phase2_options:
        options_data = phase2_options[ticker]
        history_df = phase1_data.get(ticker, {}).get("history")
        if history_df is None:
            continue

        # Temporarily build a StrategySelector to extract ATM IV from the chain
        temp_signals = signals_data.get(ticker) or {}
        if not temp_signals:
            continue

        temp_selector = StrategySelector(ticker, temp_signals, options_data, r=r)
        atm_iv = temp_selector._get_atm_iv()  # decimal (e.g. 0.35) or None

        # Re-run indicators with refined ATM IV for a more accurate IV Rank
        refined_signals = analyze_ticker(ticker, history_df, current_atm_iv=atm_iv)
        if refined_signals is None:
            refined_signals = temp_signals

        selector = StrategySelector(
            ticker, refined_signals, options_data,
            min_credit_pct=min_credit_pct, r=r,
        )
        results.extend(selector.run())

    return pd.DataFrame(results) if results else pd.DataFrame()


if __name__ == "__main__":
    # Quick gate verification with synthetic data
    print("Gate tests:")
    print(
        "  passes_bidask_gate SPY bid=1.00 ask=1.04 :",
        passes_bidask_gate(pd.Series({"bid": 1.00, "ask": 1.04}), "SPY"),
    )
    print(
        "  passes_bidask_gate AAPL bid=1.00 ask=1.12 (>10%):",
        passes_bidask_gate(pd.Series({"bid": 1.00, "ask": 1.12}), "AAPL"),
    )
    print(
        "  passes_oi_gate OI=1500 :",
        passes_oi_gate(pd.Series({"openInterest": 1500})),
    )
    print(
        "  passes_oi_gate OI=800 :",
        passes_oi_gate(pd.Series({"openInterest": 800})),
    )
    print(
        "  passes_yield_gate credit=1.20 width=5.0 :",
        passes_yield_gate(1.20, 5.0),
    )
    print(
        "  passes_yield_gate credit=0.90 width=5.0 :",
        passes_yield_gate(0.90, 5.0),
    )
    print(
        "  get_delta c S=485 K=505 iv=20% dte=35 :",
        round(get_delta("c", 485.0, 505.0, 20.0, 35), 3),
    )
    print(
        "  get_delta p S=485 K=465 iv=22% dte=35 :",
        round(get_delta("p", 485.0, 465.0, 22.0, 35), 3),
    )
