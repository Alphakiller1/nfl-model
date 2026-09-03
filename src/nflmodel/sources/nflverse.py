"""nflverse client -- schedules, weekly stats and play-level scheme sources.

The core public files require no key:

* ``nflverse/nfldata`` ``games.csv`` -- every game from 1999 to the current
  season, with the closing spread, total and both moneylines. This is where the
  2026 schedule and its market come from.
* ``nflverse-data`` ``stats_team_week_<season>.csv`` -- one row per team per
  game, carrying EPA, first downs, explosive plays, sacks and turnovers.
* play-by-play, participation and FTN charting releases -- a narrow set of
  formation, personnel, coverage and play-design fields for the scheme matrix.

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
PLAYER_WEEK_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{season}.csv"
)
WEEKLY_ROSTER_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "weekly_rosters/roster_weekly_{season}.csv"
)
DEPTH_CHART_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "depth_charts/depth_charts_{season}.csv"
)
INJURIES_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "injuries/injuries_{season}.csv"
)
PBP_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "pbp/play_by_play_{season}.csv.gz"
)
PARTICIPATION_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "pbp_participation/pbp_participation_{season}.csv"
)
FTN_CHARTING_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "ftn_charting/ftn_charting_{season}.csv"
)

CACHE_DIR = Path(os.getenv("NFL_MODEL_CACHE")
                 or Path(__file__).resolve().parents[3] / "data" / "cache")
TIMEOUT = 90
RETRIES = 3
# Volatile files. Six hours is short enough that a Tuesday line move lands on the
# next scheduled build, and long enough that a burst of local rebuilds costs one
# download rather than twenty.
VOLATILE_TTL = 6 * 3600

SCHEME_PBP_FIELDS = (
    "season", "week", "season_type", "game_id", "play_id", "posteam", "defteam",
    "play_type", "down", "ydstogo", "yardline_100", "score_differential", "wp",
    "qtr", "pass", "rush", "shotgun", "no_huddle", "pass_oe", "pass_length",
    "pass_location", "run_location", "run_gap", "epa", "success", "air_yards",
    "yards_gained", "sack", "complete_pass", "touchdown", "receiver_player_id",
)
SCHEME_PARTICIPATION_FIELDS = (
    "nflverse_game_id", "play_id", "possession_team", "offense_formation",
    "offense_personnel", "defenders_in_box", "defense_personnel",
    "number_of_pass_rushers", "time_to_throw", "was_pressure", "route",
    "defense_man_zone_type", "defense_coverage_type",
)
SCHEME_FTN_FIELDS = (
    "nflverse_game_id", "season", "week", "nflverse_play_id", "qb_location",
    "n_offense_backfield", "n_defense_box", "is_no_huddle", "is_motion",
    "is_play_action", "is_screen_pass", "is_rpo", "is_trick_play",
    "is_qb_out_of_pocket", "n_blitzers", "n_pass_rushers",
)


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
    if ttl <= 0:
        return False
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


def _parse_selected(
    raw: bytes, fields: tuple[str, ...], *, compressed: bool = False
) -> list[dict]:
    """Parse only the columns a feature family uses.

    Play-by-play is hundreds of columns wide. Materialising every cell costs far
    more memory than the scheme model itself, so this parser keeps a narrow,
    explicit data contract. A missing upstream column remains an empty value and
    is surfaced later as unavailable coverage; it is never silently imputed.
    """
    if compressed and raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    available = set(reader.fieldnames or ())
    return [
        {field: row.get(field, "") if field in available else "" for field in fields}
        for row in reader
    ]


def _parse_latest(raw: bytes, field: str, *, before: str | None = None) -> list[dict]:
    """Keep only the most recent dated snapshot, without retaining a huge history."""
    text = raw.decode("utf-8-sig", errors="replace")
    latest = ""
    selected: list[dict] = []
    for row in csv.DictReader(io.StringIO(text)):
        value = str(row.get(field) or "")
        if not value or (before is not None and value > before):
            continue
        if value > latest:
            latest = value
            selected = [row]
        elif value == latest:
            selected.append(row)
    return selected


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


def fetch_selected_csv(
    url: str,
    cache_name: str,
    *,
    ttl: float | None,
    fields: tuple[str, ...],
    compressed: bool = False,
) -> list[dict]:
    """Fetch a large CSV while retaining only selected fields in memory."""
    path = CACHE_DIR / cache_name
    state = "cached"
    error = None
    stale = False
    if _fresh(path, ttl):
        raw = path.read_bytes()
    else:
        try:
            raw = _download(url)
            state = "fresh"
        except NflverseError as exc:
            if not path.is_file() or not path.stat().st_size:
                _STATUS[cache_name] = SourceStatus(
                    cache_name, url, "error", None, None, 0, stale=True,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            raw = path.read_bytes()
            state = "bounded_snapshot"
            stale = True
            error = f"{type(exc).__name__}: {exc}"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(raw)
            temporary.replace(path)
    rows = _parse_selected(raw, fields, compressed=compressed)
    fetched_at, age = _file_state(path)
    _STATUS[cache_name] = SourceStatus(
        cache_name, url, state, fetched_at, age, len(rows), stale=stale, error=error
    )
    return rows


def fetch_latest_csv(
    url: str,
    cache_name: str,
    *,
    ttl: float | None,
    field: str,
    before: str | None = None,
) -> list[dict]:
    """Fetch a dated CSV but retain only its last point-in-time snapshot in memory."""
    path = CACHE_DIR / cache_name
    state = "cached"
    error = None
    stale = False
    if _fresh(path, ttl):
        raw = path.read_bytes()
    else:
        try:
            raw = _download(url)
            state = "fresh"
        except NflverseError as exc:
            if not path.is_file() or not path.stat().st_size:
                _STATUS[cache_name] = SourceStatus(
                    cache_name, url, "error", None, None, 0, stale=True,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            raw = path.read_bytes()
            state = "bounded_snapshot"
            stale = True
            error = f"{type(exc).__name__}: {exc}"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(raw)
            temporary.replace(path)
    rows = _parse_latest(raw, field, before=before)
    fetched_at, age = _file_state(path)
    _STATUS[cache_name] = SourceStatus(
        cache_name, url, state, fetched_at, age, len(rows), stale=stale, error=error
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


def _optional_scheme_file(
    *,
    url: str,
    cache_name: str,
    season: int,
    fields: tuple[str, ...],
    compressed: bool = False,
) -> list[dict]:
    """Load a scheme source whose current-season release may not exist yet."""
    ttl = None if season < current_season() else VOLATILE_TTL
    try:
        return fetch_selected_csv(
            url,
            cache_name,
            ttl=ttl,
            fields=fields,
            compressed=compressed,
        )
    except NflverseError:
        if season >= current_season():
            _STATUS[cache_name] = SourceStatus(
                cache_name, url, "not_published", None, None, 0,
                error=f"{season} scheme file has not been published",
            )
            return []
        raise


def play_by_play(season: int) -> list[dict]:
    """Narrow play-level context/outcomes used by the scheme matrix."""
    return _optional_scheme_file(
        url=PBP_URL.format(season=season),
        cache_name=f"play_by_play_{season}.csv.gz",
        season=season,
        fields=SCHEME_PBP_FIELDS,
        compressed=True,
    )


def participation(season: int) -> list[dict]:
    """Formation, personnel and coverage charting for one season.

    nflverse documents that participation is released after the postseason from
    2023 onward. The dashboard therefore reports its source season rather than
    presenting last season's coverage/personnel rates as live data.
    """
    return _optional_scheme_file(
        url=PARTICIPATION_URL.format(season=season),
        cache_name=f"pbp_participation_{season}.csv",
        season=season,
        fields=SCHEME_PARTICIPATION_FIELDS,
    )


def ftn_charting(season: int) -> list[dict]:
    """FTN manual charting: motion, play action, RPO, screens and blitzers."""
    return _optional_scheme_file(
        url=FTN_CHARTING_URL.format(season=season),
        cache_name=f"ftn_charting_{season}.csv",
        season=season,
        fields=SCHEME_FTN_FIELDS,
    )


def player_week(season: int, *, completed_season: bool | None = None) -> list[dict]:
    """Weekly player stats, including passing, rushing, receiving and kicking."""
    if completed_season is None:
        completed_season = season < current_season()
    ttl = None if completed_season else VOLATILE_TTL
    try:
        return fetch_csv(
            PLAYER_WEEK_URL.format(season=season),
            f"player_stats_week_{season}.csv",
            ttl=ttl,
        )
    except NflverseError:
        return []


def weekly_roster(season: int, *, week: int | None = None) -> list[dict]:
    rows = fetch_csv(
        WEEKLY_ROSTER_URL.format(season=season),
        f"roster_weekly_{season}.csv",
        ttl=VOLATILE_TTL,
    )
    available = sorted({int(number(row.get("week")) or 0) for row in rows})
    if not available:
        return []
    eligible = [candidate for candidate in available if week is None or candidate <= week]
    if not eligible:
        return []
    selected = max(eligible)
    return [row for row in rows if int(number(row.get("week")) or 0) == selected]


def depth_charts(season: int, *, before: str | None = None) -> list[dict]:
    return fetch_latest_csv(
        DEPTH_CHART_URL.format(season=season),
        f"depth_charts_{season}.csv",
        ttl=VOLATILE_TTL,
        field="dt",
        before=before,
    )


def injuries(season: int, *, week: int | None = None) -> list[dict]:
    """Latest injury designation at or before ``week``.

    The release is legitimately absent before a season's first injury report.
    That state is recorded as ``not_published`` rather than mislabelled as a
    failed or stale feed.
    """
    cache_name = f"injuries_{season}.csv"
    url = INJURIES_URL.format(season=season)
    try:
        rows = fetch_csv(url, cache_name, ttl=VOLATILE_TTL)
    except NflverseError:
        _STATUS[cache_name] = SourceStatus(
            cache_name, url, "not_published", None, None, 0,
            error="season injury report has not been published",
        )
        return []
    available = sorted({int(number(row.get("week")) or 0) for row in rows})
    eligible = [candidate for candidate in available if week is None or candidate <= week]
    if not eligible:
        return []
    selected = max(eligible)
    return [row for row in rows if int(number(row.get("week")) or 0) == selected]


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
