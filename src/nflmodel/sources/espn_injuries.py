"""Game-day status from ESPN's complete league injury feed.

nflverse publishes the official report as of its last update, often Friday's
practice report. Saturday downgrades and IR moves reach ESPN first: for week 4
of 2026 nflverse still listed Keenan Allen as Questionable while ESPN had him
Out, so the projection layer gave Indianapolis' targets to a player who was
not playing. These rows are merged over the nflverse report (later wins).

Game summaries contain a partial spotlight and are a visibly degraded fallback.
League athlete links supply IDs when the endpoint omits an explicit id field.
Statuses and clearances must be dated within the current NFL reporting week.
Current official Out rulings cannot be erased by weaker secondary statuses.
"""

from __future__ import annotations

import concurrent.futures
import re
from datetime import datetime, timedelta, timezone

from .. import teams
from . import espn_odds
from .oddsapi import normalise, team_index

SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event}"
LEAGUE_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries?limit=1000"
# What the last fetch did, published on the board so a silent failure shows.
LAST: dict = {"state": "not_run"}
STATUS = {
    "out": "Out",
    "doubtful": "Doubtful",
    "questionable": "Questionable",
    "injured reserve": "Out",
    "physically unable to perform": "Out",
    "suspension": "Out",
    "non-football injury": "Out",
    "active": "",
    "available": "",
    "healthy": "",
}


def parse(payload: dict, *, gsis_by_espn: dict[str, str], season: int, week: int) -> list[dict]:
    rows = []
    aliases = team_index()
    for block in payload.get("injuries") or []:
        team_block = block.get("team") or {}
        name = team_block.get("displayName") or block.get("displayName") or ""
        team = aliases.get(normalise(name)) or teams.canonical(team_block.get("abbreviation") or "")
        for item in block.get("injuries") or []:
            athlete = item.get("athlete") or {}
            raw_status = str(item.get("status") or "").strip().lower()
            if raw_status not in STATUS or not team:
                continue
            status = STATUS[raw_status]
            espn_id = str(athlete.get("id") or "").split(".")[0]
            if not espn_id:
                for link in athlete.get("links") or []:
                    matched = re.search(r"/id/(\d+)(?:/|$)", str(link.get("href") or ""))
                    if matched:
                        espn_id = matched.group(1)
                        break
            gsis = gsis_by_espn.get(espn_id)
            rows.append(
                {
                    "season": str(season),
                    "week": str(week),
                    "team": team,
                    "gsis_id": gsis or "",
                    "full_name": athlete.get("displayName") or "",
                    "position": (athlete.get("position") or {}).get("abbreviation") or "",
                    "report_status": status,
                    "source": "espn",
                    "source_reported_at": item.get("date"),
                    "availability_status": "Available"
                    if raw_status in {"active", "available", "healthy"}
                    else status,
                }
            )
    return rows


def week_events(season: int, week: int) -> list[dict]:
    """Every game of the week from ESPN's scoreboard, odds posted or not.

    `espn_odds.lines` keeps only games with a DraftKings quote on ESPN, which
    overnight can be none of them: the first deploy after midnight matched 0
    games and merged no injuries.
    """
    board = espn_odds._get(f"{espn_odds.SCOREBOARD}?seasontype=2&week={week}&dates={season}")
    out = []
    for event in board.get("events") or []:
        comp = (event.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): (c.get("team") or {}) for c in comp.get("competitors") or []}
        if "home" in sides and "away" in sides:
            out.append(
                {
                    "event_id": str(event.get("id") or ""),
                    "kickoff_utc": event.get("date"),
                    "home_name": sides["home"].get("displayName") or "",
                    "away_name": sides["away"].get("displayName") or "",
                }
            )
    return out


def fetch(
    needed: set[tuple[str, str]],
    roster: list[dict],
    *,
    season: int,
    week: int,
    events: list[dict] | None = None,
) -> list[dict]:
    if events is None:
        events = week_events(season, week)
    aliases = team_index()
    gsis_by_espn = {
        str(row.get("espn_id") or "").split(".")[0]: str(row.get("gsis_id") or "")
        for row in roster
        if row.get("espn_id") and row.get("gsis_id")
    }
    # The game-summary widget is a short injury spotlight, not a complete list.
    # On Nov19 it omitted Woody Marks although the league feed listed him Out.
    # The league endpoint supplies all 32 club blocks and explicit clearances.
    needed_teams = {teams.canonical(t) for matchup in needed for t in matchup}
    league_error = None
    try:
        payload = espn_odds._get(LEAGUE_INJURIES)
        source_season = payload.get("season") or {}
        source_year = (
            source_season.get("year") if isinstance(source_season, dict) else source_season
        )
        if source_year is not None and int(source_year) != season:
            raise ValueError("League injury feed belongs to another season")
        blocks = payload.get("injuries")
        if not isinstance(blocks, list) or not blocks:
            raise ValueError("League injury feed has no club blocks")
        covered = {
            aliases.get(
                normalise((b.get("team") or {}).get("displayName") or b.get("displayName") or "")
            )
            for b in blocks
        }
        if not needed_teams <= covered:
            raise ValueError("League injury feed does not cover every playing club")
        rows = parse(payload, gsis_by_espn=gsis_by_espn, season=season, week=week)
        now = datetime.now(timezone.utc)
        dates = []
        for event in events:
            try:
                dates.append(
                    datetime.fromisoformat(str(event["kickoff_utc"]).replace("Z", "+00:00"))
                )
            except (ValueError, KeyError, TypeError):
                pass
        first = min(dates) if dates else now
        week_start = (first - timedelta(days=first.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        selected = []
        for row in rows:
            if row["team"] not in needed_teams:
                continue
            try:
                reported = datetime.fromisoformat(
                    str(row["source_reported_at"]).replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                continue
            if reported.tzinfo is None or not week_start <= reported <= now:
                continue  # Do not relabel an older injury or clearance as this week's report.
            selected.append(row)
        LAST.clear()
        LAST.update(
            {
                "state": "ok",
                "feed": "complete_league_injuries",
                "source": LEAGUE_INJURIES,
                "club_blocks": len(blocks),
                "teams_covered": len(needed_teams),
                "designations": len(selected),
                "unmatched_players": sum(not r["gsis_id"] for r in selected),
                "fetched_at_utc": now.isoformat(),
                "errors": [],
            }
        )
        return selected
    except Exception as exc:
        league_error = f"complete feed: {type(exc).__name__}: {exc}"[:160]
    wanted = []
    for event in events:
        home = aliases.get(normalise(event.get("home_name") or ""))
        away = aliases.get(normalise(event.get("away_name") or ""))
        if home and away and (teams.canonical(home), teams.canonical(away)) in needed:
            wanted.append(str(event["event_id"]))

    errors: list[str] = []

    def one(event_id):
        try:
            return parse(
                espn_odds._get(SUMMARY.format(event=event_id)),
                gsis_by_espn=gsis_by_espn,
                season=season,
                week=week,
            )
        except Exception as exc:  # a missing summary leaves the nflverse report in place
            errors.append(f"{event_id}: {type(exc).__name__}: {exc}"[:160])
            return []

    out: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for rows in pool.map(one, wanted):
            out.extend(rows)
    LAST.clear()
    LAST.update(
        {
            "state": "degraded" if out else ("error" if errors else "empty"),
            "feed": "partial_game_summary_fallback",
            "scoreboard_events": len(events),
            "matched_games": len(wanted),
            "needed_games": len(needed),
            "designations": len(out),
            "unmatched_players": sum(1 for row in out if not row["gsis_id"]),
            "errors": ([league_error] if league_error else []) + errors[:4],
        }
    )
    return out


def merge(official: list[dict], fresh: list[dict]) -> list[dict]:
    """nflverse rows overridden (by team + gsis id, else team + name) by ESPN rows."""

    def key(row):
        gsis = str(row.get("gsis_id") or "").strip()
        if gsis:
            return (row.get("team"), gsis)
        return (row.get("team"), normalise(row.get("full_name") or ""))

    merged = {key(row): row for row in official}
    for row in fresh:
        base = merged.get(key(row), {})
        if (
            str(base.get("report_status") or "").lower() == "out"
            and str(row.get("report_status") or "").lower() != "out"
        ):
            # A weaker/undated secondary designation cannot silently undo the
            # current week's official ruling. A missing next-week report is
            # handled upstream, not by carrying last week's Out forward.
            merged[key(row)] = {
                **base,
                "secondary_status": row.get("availability_status"),
                "status_conflict": "Official Out retained over secondary feed",
            }
            continue
        merged[key(row)] = {**base, **row}
    return list(merged.values())
