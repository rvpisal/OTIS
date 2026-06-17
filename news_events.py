"""
News & Events module for OTIS.

Three free data sources, no new dependencies:
  1. Embedded macro event calendar (FOMC / CPI / NFP / PCE / GDP) — these
     dates are published months in advance by the Fed and BLS.
  2. Market headlines via public RSS feeds (stdlib urllib + ElementTree).
  3. Per-ticker news and next earnings date via yfinance.
"""

import logging
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime

import yfinance as yf

logger = logging.getLogger(__name__)

# ── Macro event calendar ──────────────────────────────────────────────────────
# Sources (verified 2026-06-10):
#   FOMC: federalreserve.gov published 2026 meeting calendar
#   CPI:  BLS published 2026 release schedule (8:30 AM ET)
#   NFP:  first Friday of each month (BLS Employment Situation)
#   PCE / GDP: BEA typical end-of-month cadence — marked approximate
# importance: "high" = binary market-wide volatility events; "medium" = notable.

MACRO_EVENTS = [
    # FOMC rate decisions (announcement = second day, 2:00 PM ET)
    {"date": "2026-06-17", "event": "FOMC Rate Decision", "importance": "high",
     "note": "Includes dot plot & economic projections. Press conf 2:30 PM ET."},
    {"date": "2026-07-29", "event": "FOMC Rate Decision", "importance": "high",
     "note": "Statement 2:00 PM ET, press conference 2:30 PM ET."},
    {"date": "2026-09-16", "event": "FOMC Rate Decision", "importance": "high",
     "note": "Includes dot plot & economic projections."},
    {"date": "2026-10-28", "event": "FOMC Rate Decision", "importance": "high",
     "note": "Statement 2:00 PM ET, press conference 2:30 PM ET."},
    {"date": "2026-12-09", "event": "FOMC Rate Decision", "importance": "high",
     "note": "Includes dot plot & economic projections."},
    # CPI releases (8:30 AM ET) — BLS published schedule
    {"date": "2026-06-10", "event": "CPI (May data)", "importance": "high",
     "note": "8:30 AM ET. Inflation print — moves rates, USD, and index vol."},
    {"date": "2026-07-14", "event": "CPI (June data)", "importance": "high",
     "note": "8:30 AM ET."},
    {"date": "2026-08-12", "event": "CPI (July data)", "importance": "high",
     "note": "8:30 AM ET."},
    {"date": "2026-09-11", "event": "CPI (August data)", "importance": "high",
     "note": "8:30 AM ET."},
    {"date": "2026-10-14", "event": "CPI (September data)", "importance": "high",
     "note": "8:30 AM ET."},
    {"date": "2026-11-10", "event": "CPI (October data)", "importance": "high",
     "note": "8:30 AM ET."},
    {"date": "2026-12-10", "event": "CPI (November data)", "importance": "high",
     "note": "8:30 AM ET."},
    # NFP / jobs report — first Friday, 8:30 AM ET
    {"date": "2026-07-03", "event": "Jobs Report / NFP (June)", "importance": "high",
     "note": "8:30 AM ET. Short session (July 4th observed)."},
    {"date": "2026-08-07", "event": "Jobs Report / NFP (July)", "importance": "high",
     "note": "8:30 AM ET."},
    {"date": "2026-09-04", "event": "Jobs Report / NFP (August)", "importance": "high",
     "note": "8:30 AM ET."},
    {"date": "2026-10-02", "event": "Jobs Report / NFP (September)", "importance": "high",
     "note": "8:30 AM ET."},
    {"date": "2026-11-06", "event": "Jobs Report / NFP (October)", "importance": "high",
     "note": "8:30 AM ET."},
    {"date": "2026-12-04", "event": "Jobs Report / NFP (November)", "importance": "high",
     "note": "8:30 AM ET."},
    # PCE (Fed's preferred inflation gauge) — end of month, approximate
    {"date": "2026-06-26", "event": "PCE Inflation (May)", "importance": "medium",
     "note": "Fed's preferred inflation gauge. Date approximate (±1 day)."},
    {"date": "2026-07-31", "event": "PCE Inflation (June)", "importance": "medium",
     "note": "Date approximate (±1 day)."},
    {"date": "2026-08-28", "event": "PCE Inflation (July)", "importance": "medium",
     "note": "Date approximate (±1 day)."},
    {"date": "2026-09-25", "event": "PCE Inflation (August)", "importance": "medium",
     "note": "Date approximate (±1 day)."},
    {"date": "2026-10-30", "event": "PCE Inflation (September)", "importance": "medium",
     "note": "Date approximate (±1 day)."},
    {"date": "2026-11-25", "event": "PCE Inflation (October)", "importance": "medium",
     "note": "Date approximate (±1 day)."},
    {"date": "2026-12-23", "event": "PCE Inflation (November)", "importance": "medium",
     "note": "Date approximate (±1 day)."},
    # GDP estimates — BEA, approximate
    {"date": "2026-06-25", "event": "GDP Q1 (third estimate)", "importance": "medium",
     "note": "Date approximate. Final revision — usually low surprise."},
    {"date": "2026-07-30", "event": "GDP Q2 (advance estimate)", "importance": "high",
     "note": "Date approximate. First look at Q2 growth — biggest GDP mover."},
    {"date": "2026-08-27", "event": "GDP Q2 (second estimate)", "importance": "medium",
     "note": "Date approximate."},
    {"date": "2026-10-29", "event": "GDP Q3 (advance estimate)", "importance": "high",
     "note": "Date approximate. First look at Q3 growth."},
    {"date": "2026-11-25", "event": "GDP Q3 (second estimate)", "importance": "medium",
     "note": "Date approximate."},
]


def upcoming_events(days_ahead: int = 45) -> list[dict]:
    """Macro events from today through days_ahead, sorted, with countdown."""
    today = date.today()
    horizon = today + timedelta(days=days_ahead)
    out = []
    for ev in MACRO_EVENTS:
        try:
            d = datetime.strptime(ev["date"], "%Y-%m-%d").date()
        except ValueError:
            continue
        if today <= d <= horizon:
            out.append({**ev, "days_until": (d - today).days})
    return sorted(out, key=lambda e: e["date"])


# ── Market headlines (RSS) ────────────────────────────────────────────────────

RSS_FEEDS = [
    ("Yahoo Finance", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US"),
    ("CNBC Top News", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
    ("CNBC Markets", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
]

_UA = {"User-Agent": "Mozilla/5.0 (OTIS options screener; personal/educational use)"}


def _parse_rss(xml_bytes: bytes, source: str) -> list[dict]:
    items = []
    root = ET.fromstring(xml_bytes)
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_raw = item.findtext("pubDate") or ""
        try:
            published = parsedate_to_datetime(pub_raw)
            published = published.replace(tzinfo=None)  # naive, local-agnostic sort key
        except (ValueError, TypeError):
            published = None
        if title:
            items.append({"title": title, "link": link, "source": source,
                          "published": published})
    return items


def fetch_market_headlines(limit: int = 15) -> list[dict]:
    """Aggregate market headlines from free RSS feeds.
    Each feed failure is swallowed — a dead feed never breaks the panel."""
    headlines: list[dict] = []
    for source, url in RSS_FEEDS:
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=8) as resp:
                headlines.extend(_parse_rss(resp.read(), source))
        except Exception as e:
            logger.warning(f"RSS feed failed ({source}): {e}")

    # Dedupe by normalized title, newest first
    seen: set[str] = set()
    unique = []
    for h in sorted(headlines, key=lambda x: x["published"] or datetime.min, reverse=True):
        key = h["title"].lower()[:80]
        if key not in seen:
            seen.add(key)
            unique.append(h)
    return unique[:limit]


# ── Per-ticker news + earnings (yfinance) ─────────────────────────────────────

def fetch_ticker_news(ticker: str, limit: int = 8) -> list[dict]:
    """Recent news for one ticker. Handles both old flat and new nested
    (content-wrapped) yfinance response shapes."""
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception as e:
        logger.warning(f"[{ticker}] news fetch failed: {e}")
        return []

    out = []
    for item in raw[: limit * 2]:
        content = item.get("content", item)  # new shape nests under "content"
        title = (content.get("title") or "").strip()
        if not title:
            continue
        link = ""
        click = content.get("clickThroughUrl") or content.get("canonicalUrl") or {}
        if isinstance(click, dict):
            link = click.get("url", "")
        link = link or item.get("link", "")
        provider = content.get("provider", {})
        source = provider.get("displayName", "") if isinstance(provider, dict) \
            else str(content.get("publisher", ""))
        published = None
        pub_raw = content.get("pubDate") or content.get("displayTime")
        if pub_raw:
            try:
                published = datetime.fromisoformat(str(pub_raw).replace("Z", "+00:00")) \
                    .replace(tzinfo=None)
            except ValueError:
                pass
        elif item.get("providerPublishTime"):
            try:
                published = datetime.fromtimestamp(int(item["providerPublishTime"]))
            except (ValueError, OSError):
                pass
        out.append({"title": title, "link": link, "source": source, "published": published})
        if len(out) >= limit:
            break
    return out


def fetch_next_earnings(ticker: str) -> date | None:
    """Next scheduled earnings date, or None if unavailable."""
    try:
        cal = yf.Ticker(ticker).calendar
        dates = None
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date")
        elif cal is not None and hasattr(cal, "loc") and "Earnings Date" in getattr(cal, "index", []):
            dates = list(cal.loc["Earnings Date"])
        if not dates:
            return None
        if not isinstance(dates, (list, tuple)):
            dates = [dates]
        today = date.today()
        future = []
        for d in dates:
            if isinstance(d, datetime):
                d = d.date()
            if isinstance(d, date) and d >= today:
                future.append(d)
        return min(future) if future else None
    except Exception as e:
        logger.warning(f"[{ticker}] earnings fetch failed: {e}")
        return None


def fetch_earnings_for_tickers(tickers: list[str]) -> dict[str, date | None]:
    """Earnings dates for a list of tickers (used in Phase 2 of the pipeline)."""
    return {t: fetch_next_earnings(t) for t in tickers}


if __name__ == "__main__":
    print("Upcoming macro events (45 days):")
    for ev in upcoming_events():
        flag = "🔴" if ev["importance"] == "high" else "🟡"
        print(f"  {flag} {ev['date']} (+{ev['days_until']}d) {ev['event']}")

    print("\nMarket headlines:")
    for h in fetch_market_headlines(limit=8):
        ts = h["published"].strftime("%m-%d %H:%M") if h["published"] else "?"
        print(f"  [{h['source']} {ts}] {h['title'][:90]}")

    print("\nAAPL news:")
    for n in fetch_ticker_news("AAPL", limit=4):
        print(f"  [{n['source']}] {n['title'][:90]}")

    print(f"\nAAPL next earnings: {fetch_next_earnings('AAPL')}")
