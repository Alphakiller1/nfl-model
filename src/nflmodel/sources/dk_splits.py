"""DraftKings public betting splits: % of handle vs % of bets, per side.

DraftKings Network publishes, for every game it books, the share of *tickets*
(bets) and the share of *money* (handle) on each side of the moneyline, spread
and total. When the money share on a side runs well ahead of its ticket share,
fewer, larger bets are on it - the standard public proxy for where sharp money
sits. Combined with which way the line has moved, it is the "sharp action"
signal books and touts quote.

The page is server-rendered HTML, paginated, one event group per league:

    NCAA Football  tb_eg=87637
    NFL            tb_eg=88808

Free, no key. Parsed defensively: a layout change yields no rows (reported),
never a wrong number.
"""

from __future__ import annotations

import html as html_lib
import json
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

URL = ("https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"
       "?tb_eg={group}&tb_edate=n7days&tb_emt=0&tb_page={page}")
GROUPS = {"cfb": 87637, "nfl": 88808}
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/128 Safari/537.36")
TIMEOUT = 30
MAX_PAGES = 12
FRESH_SECONDS = 20 * 60
CACHE_DIR = Path(os.environ.get(
    "DK_SPLITS_CACHE",
    str(Path(__file__).resolve().parents[3] / "data" / "cache" / "dk_splits"),
))


@dataclass(frozen=True)
class Side:
    market: str          # moneyline | spread | total
    selection: str       # "Virginia Tech", "Virginia Tech -2.5", "Over 47.5"
    odds: int | None
    handle_pct: float    # 0-100
    bets_pct: float      # 0-100


@dataclass(frozen=True)
class GameSplits:
    away: str
    home: str
    start: str           # as printed, e.g. "10/2, 07:00PM"
    sides: tuple[Side, ...]

    def to_json(self) -> dict:
        return {"away": self.away, "home": self.home, "start": self.start,
                "sides": [asdict(s) for s in self.sides]}


_TAGS = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


def _tokens(fragment: str) -> list[str]:
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", fragment, flags=re.S)
    text = _TAGS.sub("|", text)
    parts = [_SPACE.sub(" ", html_lib.unescape(t)).strip() for t in text.split("|")]
    return [p for p in parts if p]


def _odds(text: str) -> int | None:
    try:
        return int(text.replace("−", "-").replace("+", ""))
    except ValueError:
        return None


def _pct(text: str) -> float | None:
    m = re.fullmatch(r"(\d{1,3})%", text)
    return float(m.group(1)) if m else None


_MARKETS = {"Moneyline": "moneyline", "Spread": "spread", "Total": "total"}
_TITLE = re.compile(r"^(?P<away>.+?) @ (?P<home>.+)$")


def parse(page_html: str) -> list[GameSplits]:
    """Every game block on one splits page."""
    out: list[GameSplits] = []
    chunks = page_html.split('class="tb-se-title')[1:]
    for chunk in chunks:
        tokens = _tokens('<x class="tb-se-title' + chunk)
        # NFL titles are split across tags: "IND Colts @" | "WAS Commanders".
        for j in range(len(tokens) - 1):
            if tokens[j].endswith("@"):
                tokens[j] = f"{tokens[j]} {tokens[j + 1]}"
                tokens[j + 1] = ""
        tokens = [t for t in tokens if t]
        title = next((t for t in tokens if _TITLE.match(t)), None)
        if title is None:
            continue
        m = _TITLE.match(title)
        start_idx = tokens.index(title)
        start = tokens[start_idx + 1] if start_idx + 1 < len(tokens) else ""
        sides: list[Side] = []
        market = None
        i = start_idx + 1
        while i < len(tokens):
            tok = tokens[i]
            if tok in _MARKETS:
                market = _MARKETS[tok]
                i += 1
                continue
            if market and i + 3 < len(tokens):
                odds = _odds(tokens[i + 1])
                handle, bets = _pct(tokens[i + 2]), _pct(tokens[i + 3])
                if odds is not None and handle is not None and bets is not None:
                    sides.append(Side(market, tok, odds, handle, bets))
                    i += 4
                    continue
            i += 1
        if sides:
            out.append(GameSplits(m["away"].strip(), m["home"].strip(), start, tuple(sides)))
    return out


def _get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.read().decode("utf-8", errors="replace")


_STATUS: dict = {"state": "not_run"}


def status() -> dict:
    return dict(_STATUS)


def fetch(league: str) -> list[GameSplits]:
    """All games for a league across pages; a last-good snapshot on failure."""
    global _STATUS
    cache = CACHE_DIR / f"{league}.json"
    if cache.is_file() and time.time() - cache.stat().st_mtime < FRESH_SECONDS:
        data = json.loads(cache.read_text(encoding="utf-8"))
        _STATUS = {"state": "cached", "games": len(data)}
        return [_from_json(g) for g in data]
    games: dict[tuple[str, str], GameSplits] = {}
    try:
        for page in range(1, MAX_PAGES + 1):
            found = parse(_get(URL.format(group=GROUPS[league], page=page)))
            new = [g for g in found if (g.away, g.home) not in games]
            for g in found:
                games[(g.away, g.home)] = g
            if not new:
                break
    except Exception as exc:
        if cache.is_file():
            data = json.loads(cache.read_text(encoding="utf-8"))
            _STATUS = {"state": "stale", "games": len(data), "error": str(exc)}
            return [_from_json(g) for g in data]
        _STATUS = {"state": "error", "error": f"{type(exc).__name__}: {exc}"}
        return []
    rows = list(games.values())
    if rows:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([g.to_json() for g in rows]), encoding="utf-8")
    _STATUS = {"state": "fresh" if rows else "empty", "games": len(rows)}
    return rows


def _from_json(row: dict) -> GameSplits:
    return GameSplits(row["away"], row["home"], row["start"],
                      tuple(Side(**s) for s in row["sides"]))
