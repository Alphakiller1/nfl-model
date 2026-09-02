"""The Odds API client for exact-identity live NFL sportsbook lines.

The board is deliberately single-book.  A requested DraftKings quote may not
fall through to FanDuel or to whichever bookmaker happens to appear first in a
provider response.  Missing DraftKings coverage is a source state, not an
invitation to relabel another book's number.

One request carries moneyline, spread, and total.  The free ``/sports`` endpoint
preflights quota, the paid response is cached briefly, and every surfaced quote
retains both the provider event time and the bookmaker update time.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .. import teams

BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "odds"
CACHE_TTL_SECONDS = 15 * 60
TIMEOUT = 45
DEFAULT_BOOK = "draftkings"


class OddsAPIError(RuntimeError):
    """The provider response could not be used safely."""


class MissingKey(OddsAPIError):
    """No Odds API key is configured."""


class QuotaExhausted(OddsAPIError):
    """The configured quota floor would be crossed."""


@dataclass(frozen=True)
class OddsStatus:
    state: str
    requested_book: str
    fetched_at: str | None
    remaining: int | None
    events: int
    matched: int
    unmatched: int
    age_seconds: int | None = None
    stale: bool = False
    error: str | None = None


@dataclass(frozen=True)
class BookLine:
    book: str
    book_title: str
    home_spread: float | None
    total: float | None
    home_moneyline: float | None
    away_moneyline: float | None
    last_update: str | None
    commence_time: str | None

    @property
    def home_margin(self) -> float | None:
        """Expected home margin; books quote the opposite handicap sign."""
        return None if self.home_spread is None else -self.home_spread


_LAST_STATUS = OddsStatus("not_run", DEFAULT_BOOK, None, None, 0, 0, 0)


def _stamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _age_seconds(value: str | None) -> int | None:
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0, int((datetime.now(timezone.utc) - moment).total_seconds()))


def status_report() -> dict:
    return asdict(_LAST_STATUS)


def _load_key() -> str | None:
    key = os.getenv("ODDS_API_KEY")
    if key:
        return key.strip()
    env = Path(__file__).resolve().parents[3] / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("ODDS_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def _cache_path(url: str) -> Path:
    # Hashing the URL also hashes the key, so the credential never appears in a
    # path, log, or cache body.
    return CACHE_DIR / (hashlib.sha256(url.encode()).hexdigest()[:20] + ".json")


def _get(path: str, params: dict, *, ttl: int = CACHE_TTL_SECONDS) -> tuple[list | dict, dict]:
    key = _load_key()
    if not key:
        raise MissingKey("ODDS_API_KEY is not set")
    url = f"{BASE}{path}?" + urllib.parse.urlencode({**params, "apiKey": key})
    cached = _cache_path(url)
    if ttl and cached.is_file() and time.time() - cached.stat().st_mtime < ttl:
        try:
            payload = json.loads(cached.read_text(encoding="utf-8"))
            headers = dict(payload.get("headers", {}))
            headers["source"] = "cache"
            headers["fetched_at"] = payload.get("fetched_at")
            return payload["data"], headers
        except (json.JSONDecodeError, KeyError, TypeError):
            cached.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            data = json.load(response)
            headers = {
                "remaining": response.headers.get("x-requests-remaining"),
                "used": response.headers.get("x-requests-used"),
                "last_cost": response.headers.get("x-requests-last"),
                "source": "live",
                "fetched_at": _stamp(),
            }
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise OddsAPIError(f"Odds API rejected the key ({exc.code})") from exc
        if exc.code == 429:
            raise QuotaExhausted("Odds API quota exhausted (429)") from exc
        raise OddsAPIError(f"Odds API error {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise OddsAPIError(f"Odds API request failed: {exc}") from exc
    if ttl:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        temporary = cached.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({"data": data, "headers": headers, "fetched_at": headers["fetched_at"]}),
            encoding="utf-8",
        )
        temporary.replace(cached)
    return data, headers


def remaining() -> int | None:
    """Credits remaining.  The provider documents ``/sports`` as zero-cost."""
    try:
        _, headers = _get("/sports/", {}, ttl=60)
    except OddsAPIError:
        return None
    value = headers.get("remaining")
    return int(value) if value is not None else None


def normalise(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(name or ""))
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return "".join(ch for ch in ascii_only.lower() if ch.isalnum())


def team_index() -> dict[str, str]:
    """Exact provider/team aliases only; ambiguous city-only names are excluded."""
    index: dict[str, str] = {}
    for abbreviation, team in teams.TEAMS.items():
        for label in (team.name, abbreviation):
            index[normalise(label)] = abbreviation
    # Provider spellings observed historically or used after relocations.
    aliases = {
        "washingtonfootballteam": "WAS",
        "washingtonredskins": "WAS",
        "oaklandraiders": "LV",
        "sandiegochargers": "LAC",
        "stlouisrams": "LA",
    }
    index.update(aliases)
    return index


def match_team(name: str, index: dict[str, str] | None = None) -> str | None:
    return (index or team_index()).get(normalise(name))


def _pick_book(bookmakers: list[dict], requested: str) -> dict | None:
    return next((book for book in bookmakers if book.get("key") == requested), None)


def _number(value, *, low: float, high: float) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if low <= number <= high else None


def fetch_lines(
    *, book: str | None = None, min_remaining: int = 20
) -> dict[tuple[str, str], BookLine]:
    """Return ``(home_abbr, away_abbr) -> BookLine`` for exactly one book."""
    global _LAST_STATUS
    requested = (os.getenv("ODDS_BOOKMAKERS") or book or DEFAULT_BOOK).strip().lower()
    if "," in requested or not requested:
        raise OddsAPIError("NFL production accepts exactly one sportsbook")
    left = remaining()
    if left is not None and left < min_remaining:
        _LAST_STATUS = OddsStatus(
            "quota_floor",
            requested,
            None,
            left,
            0,
            0,
            0,
            error=f"only {left} credits left (floor {min_remaining})",
        )
        raise QuotaExhausted(f"only {left} Odds API credits left (floor {min_remaining})")
    query = {
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "bookmakers": requested,
    }
    try:
        data, headers = _get(f"/sports/{SPORT}/odds", query)
    except Exception as exc:
        _LAST_STATUS = OddsStatus(
            "error", requested, None, left, 0, 0, 0, error=f"{type(exc).__name__}: {exc}"
        )
        raise

    index = team_index()
    out: dict[tuple[str, str], BookLine] = {}
    unmatched = 0
    for event in data:
        home = match_team(event.get("home_team", ""), index)
        away = match_team(event.get("away_team", ""), index)
        selected = _pick_book(event.get("bookmakers", []), requested)
        if not home or not away or not selected:
            unmatched += 1
            continue
        spread = total = home_moneyline = away_moneyline = None
        for market in selected.get("markets", []):
            key = market.get("key")
            for outcome in market.get("outcomes", []):
                outcome_team = match_team(outcome.get("name", ""), index)
                if key == "spreads" and outcome_team == home:
                    spread = _number(outcome.get("point"), low=-40.0, high=40.0)
                elif key == "totals" and str(outcome.get("name", "")).lower() == "over":
                    total = _number(outcome.get("point"), low=20.0, high=90.0)
                elif key == "h2h" and outcome_team == home:
                    home_moneyline = _number(outcome.get("price"), low=-100000.0, high=100000.0)
                elif key == "h2h" and outcome_team == away:
                    away_moneyline = _number(outcome.get("price"), low=-100000.0, high=100000.0)
        if all(value is None for value in (spread, total, home_moneyline, away_moneyline)):
            unmatched += 1
            continue
        out[(home, away)] = BookLine(
            book=selected["key"],
            book_title=selected.get("title", selected["key"]),
            home_spread=spread,
            total=total,
            home_moneyline=home_moneyline,
            away_moneyline=away_moneyline,
            last_update=selected.get("last_update"),
            commence_time=event.get("commence_time"),
        )
    _LAST_STATUS = OddsStatus(
        "fresh" if headers.get("source") == "live" else "cached",
        requested,
        headers.get("fetched_at"),
        int(headers["remaining"]) if headers.get("remaining") is not None else left,
        len(data),
        len(out),
        unmatched,
        age_seconds=_age_seconds(headers.get("fetched_at")),
    )
    return out
