"""Observed NFL thresholds from PrizePicks' public, first-party research pages.

No sportsbook lines, inferred standard lines, or fabricated milestones. The
public cards show More/Less but do not expose entry availability, variant or
payout metadata; these remain unverified. The direct API is not required.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import re
import unicodedata
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo

from .. import teams
from .chase_context import timestamp
from .oddsapi import normalise

BASE = "https://www.prizepicks.com/research/nfl/players/"
CACHE_DIR = Path(__file__).resolve().parents[3] / "data/cache/prizepicks"
MAX_BYTES = 6_000_000
CACHE_SECONDS = 15 * 60
MAX_AGE_SECONDS = 60 * 60
MARKETS = {
    "Pass Yards": ("passing_yards", "Passing yards"),
    "Pass TDs": ("passing_tds", "Passing touchdowns"),
    "Pass Attempts": ("pass_attempts", "Pass attempts"),
    "Pass ATT": ("pass_attempts", "Pass attempts"),
    "Pass Completions": ("completions", "Completions"),
    "INT": ("interceptions", "Interceptions"),
    "Interceptions": ("interceptions", "Interceptions"),
    "Rush Yards": ("rushing_yards", "Rushing yards"),
    "Rush Attempts": ("rush_attempts", "Rush attempts"),
    "Rush ATT": ("rush_attempts", "Rush attempts"),
    "Rec Yds": ("receiving_yards", "Receiving yards"),
    "Receptions": ("receptions", "Receptions"),
    "Rush+Rec Yds": ("rush_receiving_yards", "Rush + receiving yards"),
    "P + R Yds": ("pass_rush_yards", "Pass + rushing yards"),
    "FG Made": ("fg_made", "Field goals made"),
    "PAT Made": ("pat_made", "Extra points made"),
    "Kicking Points": ("kicking_points", "Kicking points"),
    "Kick Pts": ("kicking_points", "Kicking points"),
}


@dataclass(frozen=True)
class PrizePicksLine:
    projection_id: str
    player_id: str
    player_name: str
    team: str
    opponent: str
    kickoff_utc: str
    metric: str
    market: str
    source_stat: str
    line: float
    observed_at_utc: str
    source_url: str
    provider: str = "prizepicks"
    variant: str = "unverified"
    # Published card labels, not confirmed selections in a user's contest.
    published_sides: tuple[str, ...] = ("MORE", "LESS")
    entry_availability_verified: bool = False


class _Cards(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cards: list[dict] = []
        self.card = None
        self.scripts: list[str] = []
        self.in_script = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script":
            self.in_script = True
        if tag == "article" and attrs.get("aria-label", "").endswith(" projection"):
            self.card = {"label": attrs["aria-label"], "text": []}

    def handle_data(self, value):
        if self.in_script:
            self.scripts.append(value)
        elif self.card is not None and value.strip():
            self.card["text"].append(value.strip())

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False
        if tag == "article" and self.card is not None:
            self.cards.append(self.card)
            self.card = None


def slug(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")


def parse(page: str, player, *, observed_at: str, source_url: str) -> list[PrizePicksLine]:
    """Read actual cards, IDs and next-game time; fail closed on identity/layout drift."""
    parser = _Cards()
    # Historical stat tables dwarf the card region. Parse only observed cards,
    # and decode Next's JSON strings separately rather than parsing all tables.
    cards = re.findall(
        r'<article\b[^>]*aria-label="[^"]+ projection"[^>]*>.*?</article>', page, re.S
    )
    parser.feed("".join(cards))
    parser.scripts = [
        script
        for script in re.findall(r"<script\b[^>]*>(.*?)</script>", page, re.S)
        if "self.__next_f.push(" in script
    ]
    streams = []
    for script in parser.scripts:
        for match in re.finditer(r"self\.__next_f\.push\(", script):
            try:
                record, _ = json.JSONDecoder().raw_decode(script[match.end() :])
                if len(record) > 1 and record[0] == 1 and isinstance(record[1], str):
                    streams.append(record[1])
            except (ValueError, IndexError, TypeError):
                continue
    rendered = "".join(streams)
    kickoff = timestamp(player.kickoff_utc)
    observed = timestamp(observed_at)
    if kickoff is None or observed is None or kickoff <= observed:
        return []
    source_times = [
        timestamp(value) for value in re.findall(r'"startTime"\s*:\s*"([^"]+)"', rendered)
    ]
    if not any(moment and abs((moment - kickoff).total_seconds()) < 60 for moment in source_times):
        return []
    local_time = kickoff.astimezone(ZoneInfo("America/New_York"))
    card_date = (
        f"{local_time:%b} {local_time.day}, {local_time.hour % 12 or 12}:{local_time:%M %p} ET"
    )
    ids = {}
    for match in re.finditer(
        r'\["\$","article","(\d+)",\{"aria-label":("(?:\\.|[^"\\])*")', rendered
    ):
        ids[json.loads(match[2])] = match[1]
    out, seen = [], set()
    for card in parser.cards:
        match = re.fullmatch(r"(.+), (\d+(?:\.\d+)?) (.+) projection", card["label"])
        if not match or normalise(match[1]) != normalise(player.player_name):
            continue
        text = card["text"]
        if "NFL" not in text or "More" not in text or "Less" not in text:
            continue
        if card_date not in text:
            continue
        club_role = next((item for item in text if " · " in item), "")
        if " · " not in club_role:
            continue
        club, role = club_role.split(" · ", 1)
        role = "K" if role in {"PK", "K"} else role
        if teams.canonical(club) != player.team or role != player.position:
            continue
        fixture = next(
            (item for item in text if re.fullmatch(r"[A-Z]{2,3} vs [A-Z]{2,3}", item)), ""
        )
        if not fixture or {teams.canonical(team) for team in fixture.split(" vs ")} != {
            player.team,
            player.opponent,
        }:
            continue
        spec = MARKETS.get(match[3])
        projection_id = ids.get(card["label"])
        line = float(match[2])
        if not spec or not projection_id or not math.isfinite(line) or projection_id in seen:
            continue
        seen.add(projection_id)
        out.append(
            PrizePicksLine(
                projection_id,
                player.player_id,
                player.player_name,
                player.team,
                player.opponent,
                kickoff.isoformat(),
                spec[0],
                spec[1],
                match[3],
                line,
                observed_at,
                source_url,
            )
        )
    return out


def _page(url: str, *, now: datetime) -> tuple[str, str]:
    path = CACHE_DIR / (hashlib.sha256(url.encode()).hexdigest()[:20] + ".json")
    if path.is_file():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            observed = timestamp(saved["observed_at_utc"])
            if observed and 0 <= (now - observed).total_seconds() <= CACHE_SECONDS:
                return saved["page"], saved["observed_at_utc"]
        except (ValueError, KeyError, TypeError):
            pass
    request = urllib.request.Request(url, headers={"User-Agent": "nfl-model/weekly-props"})
    with urllib.request.urlopen(request, timeout=15) as response:
        raw = response.read(MAX_BYTES + 1)
        # A cached page retains the HTTP Age in its observation time.
        age = int(response.headers.get("Age") or 0)
    if len(raw) > MAX_BYTES or age > MAX_AGE_SECONDS or age < 0:
        raise ValueError("Oversized or stale PrizePicks research page")
    from datetime import timedelta

    observed = (datetime.now(UTC) - timedelta(seconds=age)).isoformat()
    page = raw.decode("utf-8")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"page": page, "observed_at_utc": observed}), encoding="utf-8")
    return page, observed


def fetch(players: list, *, now: datetime | None = None) -> tuple[list[PrizePicksLine], dict]:
    """Four polite workers, fifteen-minute cache, exact fixture and player validation."""
    moment = now or datetime.now(UTC)
    candidates = [
        p
        for p in players
        if p.position in {"RB", "QB", "WR", "K"}
        and p.depth_rank <= {"QB": 1, "K": 1, "RB": 2, "WR": 3}[p.position]
        and timestamp(p.kickoff_utc)
        and timestamp(p.kickoff_utc) > moment
    ]
    saved_path = os.getenv("NFL_PRIZEPICKS_SNAPSHOT_PATH")
    if saved_path:
        return _snapshot(Path(saved_path), candidates, moment)

    def one(player):
        url = BASE + slug(player.player_name)
        try:
            page, observed = _page(url, now=moment)
            rows = parse(page, player, observed_at=observed, source_url=url)
            return rows, None
        except (OSError, ValueError, TypeError) as exc:
            return [], f"{player.player_name}: {type(exc).__name__}"

    lines, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for rows, error in pool.map(one, candidates):
            lines.extend(rows)
            if error:
                failures.append(error)
    return lines, {
        "provider": "prizepicks",
        "state": "fresh" if lines else "unavailable",
        "source": BASE,
        "candidate_players": len(candidates),
        "matched_players": len({line.player_id for line in lines}),
        "lines": len(lines),
        "errors": failures,
        "variant": "unverified",
        "entry_availability_verified": False,
        "note": "Public PrizePicks research thresholds. Confirm variant, offered side "
        "and current line in the app; no sportsbook/milestone fallback.",
    }


def _snapshot(path: Path, players: list, now: datetime) -> tuple[list[PrizePicksLine], dict]:
    """Explicit saved first-party snapshots for deterministic/offline builds."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "nfl-prizepicks-lines/1":
            raise ValueError("Unknown PrizePicks snapshot contract")
        rows = [PrizePicksLine(**row) for row in payload["lines"]]
        ids = {p.player_id for p in players}
        rows = [
            r
            for r in rows
            if r.player_id in ids
            and r.provider == "prizepicks"
            and timestamp(r.observed_at_utc)
            and 0 <= (now - timestamp(r.observed_at_utc)).total_seconds() <= MAX_AGE_SECONDS
        ]
        return rows, {
            "provider": "prizepicks",
            "state": "fresh" if rows else "unavailable",
            "source": str(path),
            "lines": len(rows),
            "variant": "unverified",
            "entry_availability_verified": False,
        }
    except (OSError, ValueError, KeyError, TypeError):
        return [], {
            "provider": "prizepicks",
            "state": "unavailable",
            "source": str(path),
            "lines": 0,
            "errors": ["Invalid, stale or unreadable PrizePicks snapshot"],
        }


def write_snapshot(lines: list[PrizePicksLine], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"schema": "nfl-prizepicks-lines/1", "lines": [asdict(line) for line in lines]},
            indent=2,
        ),
        encoding="utf-8",
    )
