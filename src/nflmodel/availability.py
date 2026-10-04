"""Starting-quarterback availability from the injury report and the weekly roster.

The points model rates *teams*, from games in which the usual quarterback almost
always played. When he is ruled out, the rating describes a team that is not
taking the field, and `forecast._points_edge` has long named this as the main
thing the market conditions on and the model does not.

This module answers one narrow question per team: is the quarterback who has been
starting unavailable this week? It deliberately does not grade backups, weigh
other positions, or read Questionable tags. Each of those was either unmeasurable
from public data or too noisy to fit, and a guess dressed as a coefficient is
worse than an honest gap. The single adjustment it drives, `QB_OUT_POINTS`, is
fitted time-forward in `scripts/fit_availability.py`.

What counts as knowable before kickoff, and therefore what may be used:

* the week's official injury report (Out / Doubtful), published by Friday;
* the weekly roster's reserve / release statuses (IR, PUP, cut, retired, exempt).

Game-day inactives (`INA`) are announced 90 minutes before kickoff, after the
board publishes, so they are ignored. Fitting on them would be look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import teams
from .sources.nflverse import number

# Fitted in scripts/fit_availability.py: points of HOME margin lost when the home
# team's usual starting quarterback is unavailable (and gained when the away
# team's is). Time-forward evidence is in reports/availability_fit.json.
QB_OUT_POINTS = 4.185
# Points of GAME TOTAL lost per team whose usual starter is unavailable. Fitted
# the same way on totals (2020-2025, 255 flagged games): actual total minus the
# model total moves -3.13 per missing starter (se 0.82); the closing total
# moves -2.27. Time-forward on flagged games the total MAE falls 11.37 -> 10.94.
# Before this, a backup quarterback moved the margin and left the total alone.
QB_OUT_TOTAL_POINTS = -3.13
STARTER_WINDOW_GAMES = 4
UNAVAILABLE_DESIGNATIONS = frozenset({"out", "doubtful"})
# Anything that removes a player from the active 53 before game day. ACT and INA
# (game-day inactive, not knowable at publish time) are the two that do not.
UNAVAILABLE_ROSTER = frozenset({"RES", "CUT", "RET", "EXE", "SUS", "NON", "UFA", "DEV"})

EVIDENCE = "reports/availability_fit.json"


@dataclass(frozen=True)
class QuarterbackStatus:
    team: str
    starter_id: str
    starter_name: str
    attempts: float
    available: bool
    reason: str | None = None

    def to_json(self) -> dict:
        return {
            "team": self.team,
            "starter_id": self.starter_id,
            "starter_name": self.starter_name,
            "available": self.available,
            "reason": self.reason,
        }


def _team(row: dict) -> str:
    return teams.canonical(str(row.get("team") or row.get("recent_team") or ""))


def usual_starters(player_rows: list[dict], season: int, week: int, *,
                   window: int = STARTER_WINDOW_GAMES) -> dict[str, tuple[str, str, float]]:
    """Team -> (gsis id, name, attempts) of its most-used passer in its last games.

    Only games before ``week`` in ``season`` count. With none (week 1), the prior
    season's final games stand in; the roster check in `quarterback_status`
    catches a starter who has since left.
    """
    def collect(target_season: int, before_week: int | None) -> dict[str, dict]:
        by_team: dict[str, dict[int, dict[str, tuple[str, float]]]] = {}
        for row in player_rows:
            if int(number(row.get("season")) or 0) != target_season:
                continue
            if str(row.get("season_type") or "REG").upper() != "REG":
                continue
            if str(row.get("position") or "").upper() != "QB":
                continue
            row_week = int(number(row.get("week")) or 0)
            if before_week is not None and row_week >= before_week:
                continue
            attempts = float(number(row.get("attempts")) or 0.0)
            if attempts <= 0:
                continue
            team = _team(row)
            player_id = str(row.get("player_id") or "")
            if not team or not player_id:
                continue
            name = str(row.get("player_display_name") or row.get("player_name") or player_id)
            games = by_team.setdefault(team, {})
            game = games.setdefault(row_week, {})
            prev = game.get(player_id, (name, 0.0))[1]
            game[player_id] = (name, prev + attempts)
        return by_team

    current = collect(season, week)
    prior = collect(season - 1, None)
    out: dict[str, tuple[str, str, float]] = {}
    for team in set(current) | set(prior):
        games = current.get(team) or prior.get(team) or {}
        recent = sorted(games)[-window:]
        totals: dict[str, tuple[str, float]] = {}
        for game_week in recent:
            for player_id, (name, attempts) in games[game_week].items():
                totals[player_id] = (name, totals.get(player_id, (name, 0.0))[1] + attempts)
        if totals:
            player_id, (name, attempts) = max(totals.items(), key=lambda kv: kv[1][1])
            out[team] = (player_id, name, attempts)
    return out


def quarterback_status(starters: dict[str, tuple[str, str, float]],
                       injuries: list[dict], roster: list[dict]) -> dict[str, QuarterbackStatus]:
    """Is each team's usual starter unavailable, and why."""
    designation: dict[tuple[str, str], str] = {}
    for row in injuries:
        status = str(row.get("report_status") or "").strip()
        if status:
            designation[(_team(row), str(row.get("gsis_id") or ""))] = status

    roster_status: dict[str, tuple[str, str]] = {}
    roster_teams: set[str] = set()
    for row in roster:
        team = _team(row)
        roster_teams.add(team)
        player_id = str(row.get("gsis_id") or "")
        if player_id:
            roster_status[player_id] = (team, str(row.get("status") or "").upper())

    out: dict[str, QuarterbackStatus] = {}
    for team, (player_id, name, attempts) in starters.items():
        reason = None
        tag = designation.get((team, player_id))
        if tag and tag.lower() in UNAVAILABLE_DESIGNATIONS:
            reason = f"injury report: {tag}"
        elif team in roster_teams:
            on_roster = roster_status.get(player_id)
            if on_roster is None or on_roster[0] != team:
                reason = "no longer on the roster"
            elif on_roster[1] in UNAVAILABLE_ROSTER:
                reason = f"roster status {on_roster[1]}"
        out[team] = QuarterbackStatus(team, player_id, name, attempts,
                                      available=reason is None, reason=reason)
    return out


def out_flag(status: dict[str, QuarterbackStatus], team: str) -> float:
    entry = status.get(team)
    return 0.0 if entry is None or entry.available else 1.0


def total_adjustment(status: dict[str, QuarterbackStatus], home: str, away: str) -> float:
    """Change to the expected game total from quarterback availability."""
    return QB_OUT_TOTAL_POINTS * (out_flag(status, home) + out_flag(status, away))


def margin_adjustment(status: dict[str, QuarterbackStatus], home: str, away: str, *,
                      points: float | None = None) -> float:
    """Change to the expected HOME margin from quarterback availability."""
    value = QB_OUT_POINTS if points is None else points
    return value * (out_flag(status, away) - out_flag(status, home))
