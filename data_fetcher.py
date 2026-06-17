import random
import time
import logging
from datetime import datetime, date

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TICKER_UNIVERSE = [
    # Broad market + sector ETFs
    "SPY", "QQQ", "IWM", "DIA", "GLD", "TLT",
    "XLE", "XLF", "XLK", "XLV", "XLU", "XLI", "XLP",
    "GDX", "EEM", "HYG", "EFA", "USO",
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "GOOGL", "META",
    "NFLX", "AVGO", "QCOM", "INTC", "MU", "CRM", "NOW", "PANW",
    # Financials
    "JPM", "GS", "BAC", "WFC", "MS", "C", "BLK", "AXP", "COF", "SCHW",
    # Healthcare
    "JNJ", "PFE", "ABBV", "UNH", "LLY", "BMY", "MRK",
    # Consumer & Retail
    "WMT", "COST", "HD", "NKE", "MCD", "TGT", "SBUX",
    # Energy
    "XOM", "CVX", "OXY", "COP", "SLB",
    # Industrials & Aerospace
    "CAT", "BA", "RTX", "HON", "GE", "DE",
    # Communication & Media
    "T", "VZ", "DIS", "CMCSA",
    # Other high-liquidity
    "PYPL", "UBER", "COIN", "MRVL", "ORCL",
]

INDEX_ETFS = {
    "SPY", "QQQ", "IWM", "DIA", "GLD", "TLT",
    "XLE", "XLF", "XLK", "XLV", "XLU", "XLI", "XLP",
    "GDX", "EEM", "HYG", "EFA", "USO",
}

# Price floor only: sub-$30 names have coarse strikes and tiny premiums, so
# credit spreads are structurally unattractive. There is NO upper cap — the
# per-contract liquidity gates (bid/ask, OI, min bid) measure option-market
# quality directly, and mega-caps like META/LLY above $500 are highly liquid.
PRICE_MIN = 30.0
# Wide DTE window: captures front months (15-30 DTE) for calendars
# AND back months (45-75 DTE), as well as the 30-45 DTE credit spread range.
DTE_MIN = 14
DTE_MAX = 75


def is_standard_monthly(date_str: str) -> bool:
    """True if the date is a standard monthly expiration.

    Covers two cases:
    - Normal: 3rd Friday of the month (weekday 4, day 15–21).
    - Holiday-adjusted: Thursday before the 3rd Friday when that Friday
      is a US market holiday (e.g. Juneteenth on June 19).
    """
    from datetime import timedelta
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        # Normal 3rd Friday
        if d.weekday() == 4 and 15 <= d.day <= 21:
            return True
        # Thursday whose next day is the 3rd Friday (holiday-adjusted monthly)
        if d.weekday() == 3 and 14 <= d.day <= 20:
            tomorrow = d + timedelta(days=1)
            if tomorrow.weekday() == 4 and 15 <= tomorrow.day <= 21:
                return True
        return False
    except ValueError:
        return False


# Keep old name for backwards compatibility within the module
is_third_friday = is_standard_monthly


def _sleep() -> None:
    time.sleep(random.uniform(1.0, 3.0))


class DataFetcher:
    def __init__(
        self,
        tickers: list | None = None,
        history_period: str = "2y",
        enforce_price_gate: bool = True,
    ):
        self.tickers = tickers if tickers is not None else TICKER_UNIVERSE
        self.history_period = history_period
        # The $30–$500 gate keeps the watchlist screen liquid, but manual
        # single-ticker lookups should work for any symbol (e.g. META > $500).
        self.enforce_price_gate = enforce_price_gate

    def fetch_price_history(self, ticker: str) -> pd.DataFrame | None:
        _sleep()
        try:
            logger.info(f"Fetching price history for {ticker}...")
            df = yf.Ticker(ticker).history(period=self.history_period)
            if df is None or df.empty:
                logger.warning(f"[SKIP] {ticker}: empty price history")
                return None
            # yfinance can return a partial intraday row with NaN OHLC for the
            # current session — a NaN last close poisons every downstream
            # calculation (strike selection anchors to current_price).
            df = df.dropna(subset=["Close"])
            if df.empty:
                logger.warning(f"[SKIP] {ticker}: no valid closes in history")
                return None
            if self.enforce_price_gate:
                last_close = float(df["Close"].iloc[-1])
                if last_close < PRICE_MIN:
                    logger.warning(
                        f"[SKIP] {ticker}: price ${last_close:.2f} below "
                        f"${PRICE_MIN:.0f} floor"
                    )
                    return None
            return df
        except Exception as e:
            logger.error(f"[ERROR] {ticker} fetch_price_history: {e}")
            return None

    def fetch_options_chain(self, ticker: str) -> dict | None:
        _sleep()
        try:
            logger.info(f"Fetching options chain for {ticker}...")
            t = yf.Ticker(ticker)
            expirations = t.options
            if not expirations:
                logger.warning(f"[SKIP] {ticker}: no options expirations available")
                return None

            today = date.today()
            all_calls: list[pd.DataFrame] = []
            all_puts: list[pd.DataFrame] = []

            for exp_str in expirations:
                if not is_standard_monthly(exp_str):
                    continue
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if not (DTE_MIN <= dte <= DTE_MAX):
                    continue

                chain = t.option_chain(exp_str)
                calls_df = chain.calls.copy()
                puts_df = chain.puts.copy()

                for df in (calls_df, puts_df):
                    df["expiration"] = exp_str
                    df["dte"] = dte
                    # yfinance returns impliedVolatility as a decimal (0.35 = 35%)
                    df["iv_pct"] = df["impliedVolatility"] * 100.0

                all_calls.append(calls_df)
                all_puts.append(puts_df)

            if not all_calls and not all_puts:
                logger.warning(
                    f"[SKIP] {ticker}: no standard monthly expirations in "
                    f"{DTE_MIN}–{DTE_MAX} DTE window"
                )
                return None

            return {
                "calls": pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame(),
                "puts": pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame(),
            }
        except Exception as e:
            logger.error(f"[ERROR] {ticker} fetch_options_chain: {e}")
            return None

    def fetch_all_phase1(self, progress_callback=None) -> dict:
        """Phase 1: fetch price history for all tickers (no options chains yet)."""
        result: dict = {}
        total = len(self.tickers)
        for i, ticker in enumerate(self.tickers):
            history = self.fetch_price_history(ticker)
            if history is not None:
                result[ticker] = {
                    "history": history,
                    "current_price": float(history["Close"].iloc[-1]),
                }
            if progress_callback:
                progress_callback(i + 1, total, ticker)
        logger.info(
            f"Phase 1 complete: {len(result)}/{total} tickers passed all gates"
        )
        return result

    def fetch_options_for_triggered(
        self, triggered_tickers: list, progress_callback=None
    ) -> dict:
        """Phase 2: fetch options chains only for technically triggered tickers."""
        result: dict = {}
        total = len(triggered_tickers)
        for i, ticker in enumerate(triggered_tickers):
            result[ticker] = self.fetch_options_chain(ticker)
            if progress_callback:
                progress_callback(i + 1, total, ticker)
        logger.info(f"Phase 2 complete: fetched options for {total} triggered tickers")
        return result


if __name__ == "__main__":
    fetcher = DataFetcher(tickers=["SPY", "AAPL"])
    phase1 = fetcher.fetch_all_phase1()
    for tkr, data in phase1.items():
        print(f"{tkr}: {len(data['history'])} rows, last close ${data['current_price']:.2f}")
    opts = fetcher.fetch_options_for_triggered(list(phase1.keys()))
    for tkr, chain in opts.items():
        if chain:
            print(f"{tkr}: {len(chain['calls'])} call rows, {len(chain['puts'])} put rows")
        else:
            print(f"{tkr}: no qualifying options")
