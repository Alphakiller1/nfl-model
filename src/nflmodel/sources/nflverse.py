"""nflverse client -- schedules, market lines and weekly team box scores.

Two public files carry everything this repo needs, and neither requires a key:

* ``nflverse/nfldata`` ``games.csv`` -- every game from 1999 to the current
  season, with the closing spread, total and both moneylines. This is where the
  2026 schedule and its market come from.
* ``nflverse-data`` ``stats_team_week_<season>.csv`` -- one row per team per
  game, carrying EPA, first downs, explosive plays, sacks and turnovers.

Defence is not a separate feed. Every game contributes two rows, so a team's
defensive line is its opponent's offensive line in the same game -- exact rather
than reconstructed, which is why `efficiency.py` can pair them on ``game_id``
alone.

**Caching.** A completed season never changes and is cached forever. The current
season, and ``games.csv`` (which gains scores and line moves through the week),
carry a short TTL. Getting that distinction backwards either re-downloads ten
seasons on every build or serves a stale slate, and the second failure is silent.

No third-party dependencies: ``urllib`` plus ``csv`` from the stdlib, so the
package imports anywhere without a build step -- the same guarantee cfb-model
makes about CFBD.
"""

from __future__ import annotations

import csv
import gzip
import io
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
TEAM_WEEK_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_team/stats_team_week_{season}.csv"
)

CACHE_DIR = Path(os.getenv("NFL_MODEL_CACHE")
                 or Path(__file__).resolve().parents[3] / "data" / "cache")
TIMEOUT = 90
RETRIES = 3
# Volatile files. Six hours is short enough that a Tuesday line move lands on the
# next scheduled build, and long enough that a burst of local rebuilds costs one
# download rather than twenty.
VOLATILE_TTL = 6 * 3600


@dataclass(frozen=True)
class SourceStatus:
    cache_name: str
    url: str
    state: str
    fetched_at: str | None
    age_seconds: int | None
    rows: int
    stale: bool = False
    error: str | None = None


_STATUS: dict[str, SourceStatus] = {}


class NflverseError(RuntimeError):
    """A source file could not be retrieved."""


def clear_run_state() -> None:
    _STATUS.clear()


def status_report() -> list[dict]:
    return [asdict(status) for status in _STATUS.values()]


def _file_state(path: Path) -> tuple[str | None, int | None]:
    if not path.is_file():
        return None, None
    modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    return (
        modified.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        max(0, int(time.time() - path.stat().st_mtime)),
    )


def _fresh(path: Path, ttl: float | None) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if ttl is None:
        return True
    return (time.time() - path.stat().st_mtime) < ttl


def _download(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "chase-analytics-nfl-model/0.2",
                         "Accept-Encoding": "gzip"},
            )
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                raw = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return raw
        except (urllib.error.URLError, OSError, gzip.BadGzipFile) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise NflverseError(f"could not fetch {url}: {last}")


def _parse(raw: bytes) -> list[dict]:
    text = raw.decode("utf-8-sig", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def fetch_csv(url: str, cache_name: str, *, ttl: float | None) -> list[dict]:
    """Fetch a CSV, honouring the cache, and return it as a list of dicts.

    A failed refresh of a file that is merely *stale* falls back to the cached
    copy rather than raising. A board built on yesterday's download is far better
    than a deploy that fails because GitHub was briefly unreachable.
    """
    path = CACHE_DIR / cache_name
    if _fresh(path, ttl):
        rows = _parse(path.read_bytes())
        fetched_at, age = _file_state(path)
        _STATUS[cache_name] = SourceStatus(
            cache_name, url, "cached", fetched_at, age, len(rows)
        )
        return rows
    try:
        raw = _download(url)
    except NflverseError as exc:
        if path.is_file() and path.stat().st_size:
            rows = _parse(path.read_bytes())
            fetched_at, age = _file_state(path)
            _STATUS[cache_name] = SourceStatus(
                cache_name,
                url,
                "bounded_snapshot",
                fetched_at,
                age,
                len(rows),
                stale=True,
                error=f"{type(exc).__name__}: {exc}",
            )
            return rows
        _STATUS[cache_name] = SourceStatus(
            cache_name, url, "error", None, None, 0, stale=True,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    rows = _parse(raw)
    fetched_at, age = _file_state(path)
    _STATUS[cache_name] = SourceStatus(
        cache_name, url, "fresh", fetched_at, age, len(rows)
    )
    return rows


def number(value) -> float | None:
    """CSV cell -> float, treating blanks and NA markers as missing.

    Missing must stay ``None``. A blank score coerced to 0.0 turns an unplayed
    game into a 0-0 result and quietly poisons every rating downstream.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "NA", "NaN", "nan", "None", "null", "N/A"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def current_season() -> int:
    """The season a date belongs to. A new NFL league year starts in March."""
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 3 else now.year - 1


def games(*, seasons: tuple[int, ...] | None = None,
          ttl: float | None = VOLATILE_TTL) -> list[dict]:
    """Every game, typed. ``seasons`` filters; ``None`` returns all of them."""
    out: list[dict] = []
    for row in fetch_csv(GAMES_URL, "games.csv", ttl=ttl):
        season = number(row.get("season"))
        if season is None:
            continue
        season = int(season)
        if seasons is not None and season not in seasons:
            continue
        week = number(row.get("week"))
        out.append({
            "game_id": row.get("game_id") or "",
            "season": season,
            "week": int(week) if week is not None else 0,
            "game_type": (row.get("game_type") or "").upper(),
            "gameday": row.get("gameday") or "",
            "weekday": row.get("weekday") or "",
            "gametime": row.get("gametime") or "",
            "away_team": row.get("away_team") or "",
            "home_team": row.get("home_team") or "",
            "away_score": number(row.get("away_score")),
            "home_score": number(row.get("home_score")),
            "location": row.get("location") or "Home",
            # nflverse quotes `spread_line` from the HOME perspective already:
            # positive means the home team is favoured by that many points, which
            # is this repo's expected-home-margin convention. It is NOT the
            # posted handicap sign, and negating it is the classic way to get
            # every forecast backwards.
            "spread_line": number(row.get("spread_line")),
            "total_line": number(row.get("total_line")),
            "home_moneyline": number(row.get("home_moneyline")),
            "away_moneyline": number(row.get("away_moneyline")),
            "div_game": bool(number(row.get("div_game"))),
            "roof": row.get("roof") or "",
            "surface": row.get("surface") or "",
            "stadium": row.get("stadium") or "",
            "home_qb_name": row.get("home_qb_name") or "",
            "away_qb_name": row.get("away_qb_name") or "",
            "home_coach": row.get("home_coach") or "",
            "away_coach": row.get("away_coach") or "",
            "home_rest": number(row.get("home_rest")),
            "away_rest": number(row.get("away_rest")),
        })
    return out


def team_week(season: int, *, completed_season: bool | None = None) -> list[dict]:
    """Weekly team box scores for one season.

    ``completed_season`` selects the cache policy; when omitted, any season older
    than the current one is treated as complete. A season nflverse has not
    published yet returns an empty list -- a schedule exists months before week
    1, and an absent stats file there is an absence, not a failure.
    """
    if completed_season is None:
        completed_season = season < current_season()
    try:
        return fetch_csv(
            TEAM_WEEK_URL.format(season=season),
            f"stats_team_week_{season}.csv",
            ttl=None if completed_season else VOLATILE_TTL,
        )
    except NflverseError:
        return []
