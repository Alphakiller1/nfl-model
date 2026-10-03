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
from zoneinfo import ZoneInfo

from . import authority as auth_mod
from . import (
    availability,
    efficiency,
    forecast,
    matrix,
    player_props,
    preseason,
    ratings,
    scheme,
    teams,
)
from .sources import espn_injuries, espn_props, nflverse, oddsapi

# How many completed seasons of history the priors need. Three is what
# `preseason.FORM_SEASON_WEIGHTS` asks for; loading fewer silently degrades the
# prior instead of failing, so the number lives next to the loader.
HISTORY_SEASONS = 3


@dataclass(frozen=True)
class Record:
    """A team's win-loss record and scoring, for display only."""

    wins: int = 0
    losses: int = 0
    ties: int = 0
    points_for: float = 0.0
    points_against: float = 0.0
    season: int = 0

    @property
    def played(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def label(self) -> str:
        base = f"{self.wins}-{self.losses}"
        return f"{base}-{self.ties}" if self.ties else base

    @property
    def points_for_per_game(self) -> float | None:
        return self.points_for / self.played if self.played else None

    @property
    def points_against_per_game(self) -> float | None:
        return self.points_against / self.played if self.played else None

    @property
    def differential(self) -> float:
        return self.points_for - self.points_against


def build_records(games: list[ratings.Game]) -> dict[str, Record]:
    """Win-loss and scoring totals per team. Ties are counted, not dropped:
    about one NFL game a season ends level and a record that hides it is wrong."""
    acc: dict[str, dict] = {}
    for game in games:
        for team, scored, allowed in ((game.home, game.home_points, game.away_points),
                                      (game.away, game.away_points, game.home_points)):
            entry = acc.setdefault(team, {"w": 0, "l": 0, "t": 0, "pf": 0.0, "pa": 0.0,
                                          "season": game.season})
            entry["pf"] += scored
            entry["pa"] += allowed
            entry["season"] = max(entry["season"], game.season)
            if scored > allowed:
                entry["w"] += 1
            elif scored < allowed:
                entry["l"] += 1
            else:
                entry["t"] += 1
    return {team: Record(wins=v["w"], losses=v["l"], ties=v["t"], points_for=v["pf"],
                         points_against=v["pa"], season=v["season"])
            for team, v in acc.items()}


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
    player_projections: list[player_props.PlayerProjection] = field(default_factory=list)
    player_status: dict = field(default_factory=dict)
    player_results: list[dict] = field(default_factory=list)
    player_prop_quotes: list[oddsapi.PlayerPropQuote] = field(default_factory=list)
    # This week's official injury report rows (nflverse), for best-bet angles.
    injuries: list[dict] = field(default_factory=list)
    scheme_profiles: dict[str, scheme.TeamSchemeProfile] = field(default_factory=dict)
    scheme_matchups: dict[tuple[str, str], scheme.SchemeMatchup] = field(default_factory=dict)
    scheme_status: dict = field(default_factory=dict)
    # Context, not model input. A preseason board with no records on it asks the
    # reader to take a rating on faith; "SEA +9.5, 12-5 last year" is a claim they
    # can check against something they remember.
    records: dict[str, "Record"] = field(default_factory=dict)
    current_forms: dict[str, matrix.TeamForm] = field(default_factory=dict)
    live_share: dict[str, float] = field(default_factory=dict)
    prior_records: dict[str, "Record"] = field(default_factory=dict)
    source_status: list[dict] = field(default_factory=list)
    odds_status: dict = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def record_for(self, team: str) -> "Record | None":
        """This season's record once it exists, otherwise last season's."""
        current = self.records.get(team)
        if current is not None and current.played:
            return current
        return self.prior_records.get(team)

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
    for index, week in enumerate(weeks):
        games = [row for row in schedule
                 if row["season"] == season and row["week"] == week
                 and row["game_type"] == "REG"]
        unplayed = [row for row in games if row["home_score"] is None]
        if not unplayed:
            continue
        # Once Sunday is done, the board looks ahead: when the only games left
        # are on the week's final day (Monday night) and the rest are final,
        # the next week - whose lines are already posted - is the slate.
        final_day = max(str(row.get("gameday") or "") for row in games)
        tail_only = all(str(row.get("gameday") or "") == final_day for row in unplayed)
        if tail_only and len(unplayed) < len(games) and index + 1 < len(weeks):
            earlier_days = {str(row.get("gameday") or "") for row in games} - {final_day}
            if earlier_days:
                return weeks[index + 1]
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
    nflverse.clear_run_state()
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
    live_lines = (
        [line for line in
         efficiency.game_lines(nflverse.team_week(season, completed_season=False))
         if line.week < week]
        if completed else []
    )

    played: dict[str, float] = {}
    for game in completed:
        played[game.home] = played.get(game.home, 0.0) + 1
        played[game.away] = played.get(game.away, 0.0) + 1

    table = preseason.blend_ratings(prior_ratings, ratings.build(completed), played)
    live_forms = preseason.live_form(live_lines)
    forms = preseason.blend_forms(prior_forms, live_forms, played)
    # Published beside the blend: the current season alone (opponent-adjusted
    # the same way) and the share of it inside the blend, so a reader shown
    # "form" can tell model form from this season's form.
    current_forms = preseason.blend_forms({}, live_forms, played)
    live_share = {team: preseason.live_weight(played.get(team, 0.0)) for team in forms}
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

    odds_error = None
    open_games = {
        (teams.canonical(row["home_team"]), teams.canonical(row["away_team"]))
        for row in games if row.get("home_score") is None
    }
    try:
        book_lines = oddsapi.fetch_lines(needed=open_games)
    except Exception as exc:
        book_lines = {}
        odds_error = f"{type(exc).__name__}: {exc}"
    issues_early: list[str] = []
    prop_quotes: list[oddsapi.PlayerPropQuote] = []
    prop_error = None
    prop_source = "oddsapi"
    if book_lines:
        try:
            prop_quotes = oddsapi.fetch_player_props(book_lines)
        except Exception as exc:
            prop_error = f"{type(exc).__name__}: {exc}"

    player_history: list[dict] = []
    for prior_season in history_seasons:
        player_history.extend(nflverse.player_week(prior_season))
    current_player_rows: list[dict] = []
    # Every published stat line this season, including games already final in
    # the board week. The ledger grades from these; projections must not see
    # them, so only rows before the board week join the history.
    season_player_rows: list[dict] = []
    if week > 1 or any(row.get("home_score") is not None for row in games):
        season_player_rows = nflverse.player_week(season, completed_season=False)
    if week > 1:
        current_player_rows = [
            row for row in season_player_rows
            if int(nflverse.number(row.get("week")) or 0) < week
        ]
        player_history.extend(current_player_rows)
    roster = nflverse.weekly_roster(season, week=week)
    if not prop_quotes and open_games:
        # The Odds API allowance runs dry mid-season; ESPN carries DraftKings'
        # prop lines (not prices) for free and identifies players by id.
        try:
            prop_quotes = espn_props.fetch(open_games, roster)
            prop_source = "espn" if prop_quotes else prop_source
        except Exception as exc:
            espn_error = f"ESPN: {type(exc).__name__}: {exc}"
            prop_error = f"{prop_error}; {espn_error}" if prop_error else espn_error
    snap_rows: list[dict] = []
    for snap_season in (season - 1, season):
        snap_rows.extend(nflverse.snap_counts(snap_season))
    first_kickoff = min((kickoff_utc(row) for row in games if kickoff_utc(row)), default=None)
    depth = nflverse.depth_charts(season, before=first_kickoff)
    injury_rows = nflverse.injuries(season, week=week)
    if open_games:
        # Game-day downgrades and IR moves reach ESPN before the nflverse report.
        try:
            injury_rows = espn_injuries.merge(
                injury_rows,
                espn_injuries.fetch(open_games, roster, season=season, week=week))
        except Exception as exc:
            issues_early.append(f"ESPN injury feed failed: {type(exc).__name__}: {exc}")
    quarterbacks = availability.quarterback_status(
        availability.usual_starters(player_history, season, week), injury_rows, roster)

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
            book=book_lines.get((teams.canonical(row["home_team"]),
                                 teams.canonical(row["away_team"]))),
            season=season, week=week,
            kickoff=kickoff_label(row),
            kickoff_utc=kickoff_utc(row),
            availability_margin=availability.margin_adjustment(
                quarterbacks, teams.canonical(row["home_team"]),
                teams.canonical(row["away_team"])),
            qb_out=tuple(
                f"{quarterbacks[t].starter_name} ({quarterbacks[t].reason})"
                for t in (teams.canonical(row["away_team"]), teams.canonical(row["home_team"]))
                if t in quarterbacks and not quarterbacks[t].available),
            authority=authority,
        )
        for row in games
    ]


    # The prior season supplies the complete participation/coverage baseline.
    # During the season, current PBP and FTN charting join it as soon as those
    # files exist. Participation from 2023 onward is an offseason nflverse
    # release, so its source season is always shown rather than called "live".
    scheme_pbp = nflverse.play_by_play(season - 1)
    scheme_participation = nflverse.participation(season - 1)
    scheme_charting = nflverse.ftn_charting(season - 1)
    if week > 1:
        scheme_pbp.extend(nflverse.play_by_play(season))
        scheme_charting.extend(nflverse.ftn_charting(season))
    scheme_result = scheme.build(
        season=season,
        week=week,
        games=games,
        schedule=schedule,
        pbp_rows=scheme_pbp,
        participation_rows=scheme_participation,
        charting_rows=scheme_charting,
        player_positions=scheme.position_index(player_history),
    )
    player_result = player_props.project(
        season=season,
        week=week,
        games=games,
        game_projections=projections,
        roster=roster,
        depth=depth,
        injuries=injury_rows,
        history_rows=player_history,
        scheme_matchups=scheme_result.matchups,
        snap_rows=snap_rows,
        box_rates=player_props.defense_box_rates(
            scheme_pbp, scheme_charting, season=season, week=week),
    )
    odds_status = oddsapi.status_report()
    # Book coverage is measured over games still to be played: a book quotes
    # no line on a final, so late in a week "1 of 16 covered" would really be
    # one of one. Projections are built in slate order, one per game.
    open_projections = [
        p for p, row in zip(projections, games)
        if nflverse.number(row.get("home_score")) is None
    ]
    odds_status.update({
        "slate_games": len(open_projections),
        "slate_total_games": len(games),
        "slate_matched": sum(p.book_name is not None for p in open_projections),
        "slate_spreads": sum(p.book_margin is not None for p in open_projections),
        "slate_totals": sum(p.book_total is not None for p in open_projections),
        "slate_moneylines": sum(
            p.book_name is not None
            and p.home_moneyline is not None
            and p.away_moneyline is not None
            for p in open_projections
        ),
        "slate_complete": sum(
            p.book_margin is not None
            and p.book_total is not None
            and p.home_moneyline is not None
            and p.away_moneyline is not None
            for p in open_projections
        ),
        "player_prop_quotes": len(prop_quotes),
        "player_prop_source": prop_source if prop_quotes else None,
    })
    issues: list[str] = list(issues_early)
    stale = [status for status in nflverse.status_report() if status.get("stale")]
    failed = [status for status in nflverse.status_report() if status.get("state") == "error"]
    if stale:
        issues.append(f"{len(stale)} nflverse source(s) used a bounded stale snapshot")
    if failed:
        issues.append(f"{len(failed)} nflverse source(s) failed")
    if games and not player_result.projections:
        issues.append("No active offensive player or kicker projections were generated")
    if odds_error:
        issues.append(f"Live sportsbook feed failed: {odds_error}")
    if prop_error:
        issues.append(f"Player-prop feed failed: {prop_error}")
    elif open_projections and not odds_status["slate_matched"]:
        issues.append("DraftKings returned no matched lines for this slate")
    elif odds_status["slate_complete"] < len(open_projections):
        issues.append(
            f"DraftKings has an incomplete spread/total/paired-moneyline set for "
            f"{len(open_projections) - odds_status['slate_complete']} game(s)"
        )
    prior = [g for g in history if g.season == season - 1]
    return Slate(season=season, week=week, table=table, forms=forms, games=games,
                 current_forms=current_forms, live_share=live_share,
                 schedule=schedule, games_played=played, authority=authority,
                 projections=projections, records=build_records(completed),
                 player_projections=player_result.projections,
                 player_status=player_result.status,
                 player_results=season_player_rows,
                 player_prop_quotes=prop_quotes,
                 injuries=injury_rows,
                 scheme_profiles=scheme_result.profiles,
                 scheme_matchups=scheme_result.matchups,
                 scheme_status=scheme_result.status,
                 prior_records=build_records(prior),
                 source_status=nflverse.status_report(), odds_status=odds_status,
                 issues=issues)


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


def kickoff_utc(row: dict) -> str:
    """Provider-comparable ISO kickoff from nflverse's Eastern wall clock."""
    day = str(row.get("gameday") or "").strip()
    clock = str(row.get("gametime") or "").strip()
    if not day or not clock:
        return ""
    try:
        local = datetime.strptime(f"{day} {clock}", "%Y-%m-%d %H:%M").replace(
            tzinfo=ZoneInfo("America/New_York")
        )
    except ValueError:
        return ""
    return local.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
