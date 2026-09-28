"""DraftKings game lines as published on ESPN's public scoreboard.

The Odds API is the primary source, but its monthly quota can run dry mid-slate
(2 credits left on 2026-09-28), which failed every production build and froze
the public board. ESPN's scoreboard carries the same DraftKings numbers -
spread, total and both moneylines - at no cost. The board stays single-book:
a quote is used only when ESPN names DraftKings as its provider, never
whichever book happens to be listed.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
TIMEOUT = 30


# ESPN's edge refuses some agents some of the time (the same plain
# "Mozilla/5.0" passed at one hour and 403'd the next), so try a short list
# and take the first that answers.
AGENTS = (None, "curl/8.5.0", "Mozilla/5.0", "python-requests/2.32")


def _get(url: str) -> dict:
    last: Exception | None = None
    for agent in AGENTS:
        headers = {"User-Agent": agent} if agent else {}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # 403 from the edge, a timeout: try the next agent
            last = exc
    raise last if last else RuntimeError("ESPN scoreboard unavailable")


def _num(value, *, low: float, high: float) -> float | None:
    try:
        # Totals are quoted "o41.5" / "u41.5"; spreads "+2.5".
        number = float(str(value).strip().lstrip("ouOU").replace("+", ""))
    except (TypeError, ValueError):
        return None
    return number if low <= number <= high else None


def _close(block: dict | None, side: str, key: str):
    side_block = (block or {}).get(side) or {}
    return ((side_block.get("close") or {}).get(key)
            if side_block.get("close") else (side_block.get("open") or {}).get(key))


def lines(requested: str = "draftkings") -> tuple[list[dict], str]:
    """Return raw per-event quotes for the current week and the fetch time.

    Each item: home/away display names, commence time, home spread, total,
    both moneylines, and ESPN's event id.
    """
    data = _get(SCOREBOARD)
    fetched = datetime.now(timezone.utc).isoformat(timespec="seconds")
    events = list(data.get("events") or [])
    # The scoreboard sits on the current week until it is over; the next
    # week's lines are already posted, so read both.
    week = (data.get("week") or {}).get("number")
    season_type = (data.get("season") or {}).get("type") or 2
    if week:
        try:
            nxt = _get(f"{SCOREBOARD}?week={int(week) + 1}&seasontype={season_type}")
            seen = {e.get("id") for e in events}
            events += [e for e in nxt.get("events") or [] if e.get("id") not in seen]
        except Exception:
            pass
    out = []
    for event in events:
        comp = (event.get("competitions") or [{}])[0]
        odds = [o for o in (comp.get("odds") or [])
                if str(((o.get("provider") or {}).get("name") or "")).strip().lower() == requested]
        if not odds:
            continue
        quote = odds[0]
        teams = {c.get("homeAway"): (c.get("team") or {}) for c in comp.get("competitors") or []}
        if "home" not in teams or "away" not in teams:
            continue
        total = _num(_close((quote.get("total") or {}), "over", "line")
                     or quote.get("overUnder"), low=20.0, high=90.0)
        home_spread = _num(_close(quote.get("pointSpread"), "home", "line"), low=-40.0, high=40.0)
        out.append({
            "event_id": str(event.get("id") or ""),
            "home_name": teams["home"].get("displayName") or "",
            "away_name": teams["away"].get("displayName") or "",
            "commence_time": event.get("date"),
            "home_spread": home_spread,
            "total": total,
            "home_moneyline": _num(_close(quote.get("moneyline"), "home", "odds"),
                                   low=-100000.0, high=100000.0),
            "away_moneyline": _num(_close(quote.get("moneyline"), "away", "odds"),
                                   low=-100000.0, high=100000.0),
        })
    return out, fetched
