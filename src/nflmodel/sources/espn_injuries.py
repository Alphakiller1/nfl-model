"""Game-day injury designations from ESPN's game summaries.

nflverse publishes the official report as of its last update, often Friday's
practice report. Saturday downgrades and IR moves reach ESPN first: for week 4
of 2026 nflverse still listed Keenan Allen as Questionable while ESPN had him
Out, so the projection layer gave Indianapolis' targets to a player who was
not playing. These rows are merged over the nflverse report (later wins).

Players are identified by ESPN athlete id, joined to gsis through the weekly
roster's ``espn_id`` column.
"""

from __future__ import annotations

import concurrent.futures

from .. import teams
from . import espn_odds
from .oddsapi import normalise, team_index

SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event}"
# What the last fetch did, published on the board so a silent failure shows.
LAST: dict = {"state": "not_run"}
STATUS = {
    "out": "Out", "doubtful": "Doubtful", "questionable": "Questionable",
    "injured reserve": "Out", "physically unable to perform": "Out",
    "suspension": "Out", "non-football injury": "Out",
}


def parse(payload: dict, *, gsis_by_espn: dict[str, str], season: int, week: int) -> list[dict]:
    rows = []
    aliases = team_index()
    for block in payload.get("injuries") or []:
        team_block = block.get("team") or {}
        team = aliases.get(normalise(team_block.get("displayName") or "")) or teams.canonical(
            team_block.get("abbreviation") or "")
        for item in block.get("injuries") or []:
            athlete = item.get("athlete") or {}
            status = STATUS.get(str(item.get("status") or "").strip().lower())
            gsis = gsis_by_espn.get(str(athlete.get("id") or ""))
            if not status or not team:
                continue
            rows.append({
                "season": str(season), "week": str(week), "team": team, "gsis_id": gsis or "",
                "full_name": athlete.get("displayName") or "",
                "position": (athlete.get("position") or {}).get("abbreviation") or "",
                "report_status": status, "source": "espn",
            })
    return rows


def fetch(needed: set[tuple[str, str]], roster: list[dict], *, season: int, week: int,
          events: list[dict] | None = None) -> list[dict]:
    if events is None:
        events, _ = espn_odds.lines()
    aliases = team_index()
    gsis_by_espn = {
        str(row.get("espn_id") or "").split(".")[0]: str(row.get("gsis_id") or "")
        for row in roster if row.get("espn_id") and row.get("gsis_id")
    }
    wanted = []
    for event in events:
        home = aliases.get(normalise(event.get("home_name") or ""))
        away = aliases.get(normalise(event.get("away_name") or ""))
        if home and away and (teams.canonical(home), teams.canonical(away)) in needed:
            wanted.append(str(event["event_id"]))

    errors: list[str] = []

    def one(event_id):
        try:
            return parse(espn_odds._get(SUMMARY.format(event=event_id)),
                         gsis_by_espn=gsis_by_espn, season=season, week=week)
        except Exception as exc:  # a missing summary leaves the nflverse report in place
            errors.append(f"{event_id}: {type(exc).__name__}: {exc}"[:160])
            return []

    out: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for rows in pool.map(one, wanted):
            out.extend(rows)
    LAST.clear()
    LAST.update({
        "state": "ok" if out else ("error" if errors else "empty"),
        "scoreboard_events": len(events), "matched_games": len(wanted),
        "needed_games": len(needed), "designations": len(out),
        "unmatched_players": sum(1 for row in out if not row["gsis_id"]),
        "errors": errors[:5],
    })
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
        merged[key(row)] = {**base, **row}
    return list(merged.values())
