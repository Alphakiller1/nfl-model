"""Data assembly: everything a board, an export or a page needs for one week.

The CLI, the dashboard and the JSON export all need the same four things --
point-in-time ratings, point-in-time form, the week's games, and the market --
and they must all get *identical* values. Building them in three places is how a
board and its export end up disagreeing about the same game, so they are built
here once and consumed everywhere.

**Point-in-time is enforced by construction, not by discipline.** `assemble`
takes the week being forecast and only ever reads games strictly before it. There
is no code path that can hand a forecaster the result it is trying to predict,
because the filter happens before the ratings solve rather than after it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import authority as auth_mod
from . import efficiency, forecast, matrix, preseason, ratings, teams
from .sources import nflverse

# How many completed seasons of history the priors need. Three is what
# `preseason.FORM_SEASON_WEIGHTS` asks for; loading fewer silently degrades the
# prior instead of failing, so the number lives next to the loader.
HISTORY_SEASONS = 3


@dataclass
class Slate:
    """One week, fully assembled."""

    season: int
    week: int
    table: dict[str, float]
    forms: dict[str, matrix.TeamForm]
    games: list[dict]
    schedule: list[dict]
    games_played: dict[str, float]
    authority: auth_mod.Authority
    projections: list[forecast.GameProjection] = field(default_factory=list)

    @property
    def in_season(self) -> bool:
        """True once the season has produced a completed game."""
        return any(v > 0 for v in self.games_played.values())

    @property
    def rated_teams(self) -> int:
        return len(self.table)


def current_week(schedule: list[dict], season: int) -> int:
    """The first week that still has an unplayed regular-season game.

    Falls back to the last week of the season once everything is complete, so a
    February rebuild renders week 18 rather than an empty board.
    """
    weeks = sorted({row["week"] for row in schedule
                    if row["season"] == season and row["game_type"] == "REG"})
    if not weeks:
        return 1
    for week in weeks:
        unplayed = [row for row in schedule
                    if row["season"] == season and row["week"] == week
                    and row["game_type"] == "REG" and row["home_score"] is None]
        if unplayed:
            return week
    return weeks[-1]


def margin_for(table: dict[str, float], forms: dict[str, matrix.TeamForm]):
    """A callable giving the published-model margin for any matchup.

    Shared with `divisions.simulate` so the season simulation and the game board
    are driven by the same number. Two estimates of one quantity is a bug waiting
    for someone to notice the division odds disagree with the spreads.
    """
    from . import totals

    def margin(home: str, away: str, neutral: bool = False) -> float | None:
        rating_margin = ratings.projected_margin(table, home, away, neutral=neutral)
        projection = totals.project(forms.get(home), forms.get(away),
                                    rating_margin=rating_margin, neutral=neutral)
        return projection.margin

    return margin


def assemble(season: int | None = None, week: int | None = None) -> Slate:
    """Load, rate and forecast one week."""
    season = season or nflverse.current_season()
    schedule = nflverse.games()
    week = week or current_week(schedule, season)

    history_seasons = tuple(range(season - HISTORY_SEASONS, season))
    history = ratings.from_rows([r for r in schedule if r["season"] in history_seasons])
    lines: list[efficiency.GameLine] = []
    for prior_season in history_seasons:
        lines.extend(efficiency.game_lines(nflverse.team_week(prior_season)))

    prior_ratings = preseason.rating_prior(history, season)
    prior_forms = preseason.form_prior(lines, season)

    # Current season, strictly before the week being forecast.
    completed = [g for g in ratings.from_rows([r for r in schedule if r["season"] == season])
                 if g.week < week]
    live_lines = [line for line in
                  efficiency.game_lines(nflverse.team_week(season, completed_season=False))
                  if line.week < week]

    played: dict[str, float] = {}
    for game in completed:
        played[game.home] = played.get(game.home, 0.0) + 1
        played[game.away] = played.get(game.away, 0.0) + 1

    table = preseason.blend_ratings(prior_ratings, ratings.build(completed), played)
    forms = preseason.blend_forms(prior_forms, preseason.live_form(live_lines), played)
    # A franchise with no history stays OUT of the table rather than being seeded
    # at 0.0. An earlier version defaulted the 32 abbreviations in and called it
    # "surfaced as unrated", which it was not: 0.0 is a rating, and it means
    # exactly league average. Every downstream None check -- `projected_margin`,
    # the AVOID action, the skipped-game path in `divisions.build_games` -- was
    # dead code as a result, and a broken relocation alias would have produced a
    # plausible average team instead of a visible gap. There are no such
    # franchises in the modern NFL; the point is that if one appeared, the hero's
    # "N teams rated" pill would say 31 instead of quietly saying 32.

    games = [row for row in schedule
             if row["season"] == season and row["week"] == week
             and row["game_type"] == "REG"]
    games.sort(key=_kickoff_key)

    authority = auth_mod.current()
    projections = [
        forecast.project_game(
            home=teams.canonical(row["home_team"]),
            away=teams.canonical(row["away_team"]),
            team_ratings=table,
            home_form=forms.get(teams.canonical(row["home_team"])),
            away_form=forms.get(teams.canonical(row["away_team"])),
            neutral=str(row.get("location") or "Home").lower() != "home",
            market_margin=row.get("spread_line"),
            market_total=row.get("total_line"),
            home_moneyline=row.get("home_moneyline"),
            away_moneyline=row.get("away_moneyline"),
            season=season, week=week,
            kickoff=kickoff_label(row),
            authority=authority,
        )
        for row in games
    ]
    return Slate(season=season, week=week, table=table, forms=forms, games=games,
                 schedule=schedule, games_played=played, authority=authority,
                 projections=projections)


# ── kickoff formatting ───────────────────────────────────────────────────────
# nflverse publishes `gameday` and `gametime` in US Eastern, which is how every
# book and broadcaster quotes an NFL kickoff. They are treated as Eastern wall
# clock and rendered as such; converting to UTC would file a Sunday-night game
# under Monday for no benefit to anyone reading the board.
def _parse_day(row: dict) -> datetime | None:
    raw = str(row.get("gameday") or "").strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _kickoff_key(row: dict) -> tuple:
    day = _parse_day(row)
    time = str(row.get("gametime") or "")
    return ((1, 0.0, "") if day is None
            else (0, day.timestamp(), time), row.get("away_team", ""))


def signed(value: float | None, places: int = 1, *, dash: str = "--") -> str:
    """A signed number with negative zero normalised away.

    Python renders -0.04 at one decimal as "-0.0", which on a board of point
    spreads reads as a defect rather than as a pick'em. Every display of a signed
    model quantity goes through here so the rule cannot drift between the CLI and
    the dashboard.
    """
    if value is None:
        return dash
    if abs(value) < 0.5 * 10 ** -places:
        value = 0.0
    return f"{value:+.{places}f}"


def kickoff_label(row: dict) -> str:
    """`Sun Sep 13 · 1:00 PM ET` -- the day matters, a week spans five of them."""
    day = _parse_day(row)
    if day is None:
        return ""
    label = f"{day:%a} {day:%b} {day.day}"
    time = str(row.get("gametime") or "").strip()
    if not time:
        return f"{label} · time TBA"
    try:
        hour, minute = (int(part) for part in time.split(":")[:2])
    except ValueError:
        return f"{label} · time TBA"
    suffix = "AM" if hour < 12 else "PM"
    display = hour % 12 or 12
    return f"{label} · {display}:{minute:02d} {suffix} ET"
