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
_MODEL_CACHE = os.getenv("NFL_MODEL_CACHE")
CACHE_DIR = Path(
    os.getenv("NFL_ODDS_CACHE")
    or (str(Path(_MODEL_CACHE) / "odds") if _MODEL_CACHE else "")
    or Path(__file__).resolve().parents[3] / "data" / "cache" / "odds"
)
CACHE_TTL_SECONDS = 15 * 60
TIMEOUT = 45
DEFAULT_BOOK = "draftkings"
# Props cost markets x regions credits per game - 12 here, ~192 for a full slate.
# Pull only inside this window before kickoff and reuse a pull this long, so a
# week costs one or two slate pulls rather than one per build.
PROP_LEAD_HOURS = 30
PROP_TTL_SECONDS = 20 * 60 * 60
PLAYER_PROP_MARKETS = (
    "player_pass_attempts",
    "player_pass_yds",
    "player_pass_tds",
    "player_rush_attempts",
    "player_rush_yds",
    "player_receptions",
    "player_reception_yds",
    "player_rush_reception_yds",
    "player_pass_rush_yds",
    "player_field_goals",
    "player_pats",
    "player_kicking_points",
)


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
    event_id: str = ""

    @property
    def home_margin(self) -> float | None:
        """Expected home margin; books quote the opposite handicap sign."""
        return None if self.home_spread is None else -self.home_spread


@dataclass(frozen=True)
class PlayerPropQuote:
    event_id: str
    home_team: str
    away_team: str
    market: str
    player_name: str
    line: float
    over_price: float
    under_price: float
    book: str
    book_title: str
    last_update: str | None
    # Set by sources that identify the player directly (ESPN athlete id ->
    # gsis); the Odds API names players only, and is matched by name.
    player_id: str | None = None
    open_line: float | None = None
    # False when the source publishes the line but not the prices (ESPN); the
    # prices above are then a nominal -110 each way.
    priced: bool = True


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


def _espn_fallback(
    requested: str, left: int | None, reason: str
) -> dict[tuple[str, str], BookLine]:
    """The same book's lines from ESPN's scoreboard when the Odds API cannot serve.

    Still exactly one book: a quote is kept only when ESPN names the requested
    book as its provider. Returns an empty map when ESPN has nothing usable,
    so the caller's original failure stands.
    """
    global _LAST_STATUS
    from . import espn_odds

    try:
        quotes, fetched = espn_odds.lines(requested)
    except Exception:
        return {}
    index = team_index()
    out: dict[tuple[str, str], BookLine] = {}
    unmatched = 0
    for q in quotes:
        home = match_team(q["home_name"], index)
        away = match_team(q["away_name"], index)
        if not home or not away:
            unmatched += 1
            continue
        out[(home, away)] = BookLine(
            event_id=f"espn:{q['event_id']}",
            book=requested,
            book_title="DraftKings" if requested == "draftkings" else requested,
            home_spread=q["home_spread"],
            total=q["total"],
            home_moneyline=q["home_moneyline"],
            away_moneyline=q["away_moneyline"],
            last_update=fetched,
            commence_time=q["commence_time"],
        )
    if not out:
        return {}
    _LAST_STATUS = OddsStatus(
        "fresh", requested, fetched, left, len(quotes), len(out), unmatched,
        age_seconds=0, error=f"{reason}; {requested} lines read from ESPN's scoreboard",
    )
    return out


def fetch_lines(
    *, book: str | None = None, min_remaining: int = 20,
    needed: set[tuple[str, str]] | None = None,
) -> dict[tuple[str, str], BookLine]:
    """Return ``(home_abbr, away_abbr) -> BookLine`` for exactly one book.

    ``needed`` is the slate's open games. When ESPN's scoreboard already carries
    the requested book's complete line for every one of them, those lines are
    used and no paid request is made: game lines are free there, and the Odds
    API allowance is kept for player props, which only it prices.
    """
    global _LAST_STATUS
    requested = (os.getenv("ODDS_BOOKMAKERS") or book or DEFAULT_BOOK).strip().lower()
    if "," in requested or not requested:
        raise OddsAPIError("NFL production accepts exactly one sportsbook")
    free_first = os.getenv("NFL_GAME_LINES_FREE_FIRST", "1").lower() not in {"0", "false", "no"}
    if needed and free_first:
        espn = _espn_fallback(requested, None, "free first")
        complete = {
            key for key, line in espn.items()
            if None not in (
                line.home_spread, line.total, line.home_moneyline, line.away_moneyline
            )
        }
        if set(needed) <= complete:
            return espn
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
        fallback = _espn_fallback(requested, left, f"Odds API at {left} credits")
        if fallback:
            return fallback
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
        fallback = _espn_fallback(requested, left, f"Odds API {type(exc).__name__}")
        if fallback:
            return fallback
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
            event_id=str(event.get("id") or ""),
            book=selected["key"],
            book_title=selected.get("title", selected["key"]),
            home_spread=spread,
            total=total,
            home_moneyline=home_moneyline,
            away_moneyline=away_moneyline,
            last_update=selected.get("last_update"),
            commence_time=event.get("commence_time"),
        )
    status = OddsStatus(
        "fresh" if headers.get("source") == "live" else "cached",
        requested,
        headers.get("fetched_at"),
        int(headers["remaining"]) if headers.get("remaining") is not None else left,
        len(data),
        len(out),
        unmatched,
        age_seconds=_age_seconds(headers.get("fetched_at")),
    )
    # A game the provider missed, or quoted without all four numbers, takes the
    # same book's full quote from ESPN (free), so one late-posting game does not
    # hold the whole board back.
    espn = _espn_fallback(requested, status.remaining, "gap fill")
    filled = 0
    for key, line in espn.items():
        have = out.get(key)
        if have is None or None in (
            have.home_spread, have.total, have.home_moneyline, have.away_moneyline
        ):
            out[key] = line
            filled += 1
    _LAST_STATUS = status if not filled else OddsStatus(
        status.state, status.requested_book, status.fetched_at, status.remaining,
        status.events, len(out), status.unmatched, age_seconds=status.age_seconds,
        error=f"{filled} game(s) filled from ESPN's {requested} lines",
    )
    return out


def _parse_player_props(payload: dict, requested: str) -> list[PlayerPropQuote]:
    """Parse paired DraftKings over/unders from one event response."""
    index = team_index()
    home = match_team(payload.get("home_team", ""), index)
    away = match_team(payload.get("away_team", ""), index)
    selected = _pick_book(payload.get("bookmakers", []), requested)
    if not home or not away or not selected:
        return []
    output: list[PlayerPropQuote] = []
    for market in selected.get("markets", []):
        key = str(market.get("key") or "")
        if key not in PLAYER_PROP_MARKETS:
            continue
        paired: dict[tuple[str, float], dict[str, float]] = {}
        for outcome in market.get("outcomes", []):
            side = str(outcome.get("name") or "").strip().lower()
            player = str(outcome.get("description") or "").strip()
            line = _number(outcome.get("point"), low=-1.0, high=1000.0)
            price = _number(outcome.get("price"), low=-100000.0, high=100000.0)
            if side not in {"over", "under"} or not player or line is None or price is None:
                continue
            paired.setdefault((player, line), {})[side] = price
        for (player, line), sides in paired.items():
            if "over" not in sides or "under" not in sides:
                continue
            output.append(PlayerPropQuote(
                event_id=str(payload.get("id") or ""),
                home_team=home,
                away_team=away,
                market=key,
                player_name=player,
                line=line,
                over_price=sides["over"],
                under_price=sides["under"],
                book=selected["key"],
                book_title=selected.get("title", selected["key"]),
                last_update=market.get("last_update"),
            ))
    return output


def fetch_player_props(
    lines: dict[tuple[str, str], BookLine],
    *,
    book: str | None = None,
    min_remaining: int = 80,
) -> list[PlayerPropQuote]:
    """Fetch paired NFL player props one event at a time.

    The provider requires the event-odds endpoint for props. Requests are made
    only for slate events already matched to the exact requested sportsbook and
    stop before crossing the configured quota floor.
    """
    requested = (os.getenv("ODDS_BOOKMAKERS") or book or DEFAULT_BOOK).strip().lower()
    if "," in requested or not requested:
        raise OddsAPIError("NFL production accepts exactly one sportsbook")
    output: list[PlayerPropQuote] = []
    left = remaining()
    # Game lines may come from ESPN, whose event ids mean nothing to this
    # provider (a 422 that took every prop down). Resolve this provider's own
    # ids from the free /events listing, by team.
    try:
        events, _ = _get(f"/sports/{SPORT}/events", {})
    except OddsAPIError:
        events = []
    event_ids = {}
    for event in events if isinstance(events, list) else []:
        home = match_team(event.get("home_team", ""))
        away = match_team(event.get("away_team", ""))
        if home and away and event.get("id"):
            event_ids[(home, away)] = str(event["id"])
    failures = 0
    now = datetime.now(timezone.utc)
    for (home, away), line in lines.items():
        event_id = event_ids.get((home, away))
        if not event_id:
            continue
        try:
            kickoff = datetime.fromisoformat(str(line.commence_time).replace("Z", "+00:00"))
        except ValueError:
            kickoff = None
        if kickoff is None or not 0 < (kickoff - now).total_seconds() < PROP_LEAD_HOURS * 3600:
            continue
        if left is not None and left < min_remaining:
            break
        query = {
            "regions": "us",
            "markets": ",".join(PLAYER_PROP_MARKETS),
            "oddsFormat": "american",
            "bookmakers": requested,
        }
        try:
            payload, headers = _get(f"/sports/{SPORT}/events/{event_id}/odds", query,
                                    ttl=PROP_TTL_SECONDS)
        except QuotaExhausted:
            raise
        except OddsAPIError:
            # One unpriceable event must not take every other game's props down.
            failures += 1
            if failures >= 3:
                raise
            continue
        if headers.get("remaining") is not None:
            left = int(headers["remaining"])
        if isinstance(payload, dict):
            output.extend(_parse_player_props(payload, requested))
    return output
