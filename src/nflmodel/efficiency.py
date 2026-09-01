"""Per-game team efficiency, and the opponent adjustment that makes it comparable.

Five offensive rates are derived from the nflverse weekly team box score, and
every one of them is a per-play quantity rather than a per-game total. A team
that trailed all afternoon runs 78 plays and a team that led runs 55; comparing
their yardage totals measures game script, not quality.

    epa_per_play      (passing_epa + rushing_epa) / plays
    first_down_rate   drive-sustaining plays / plays -- the box-score stand-in
                      for success rate, which needs play-by-play this repo does
                      not download
    explosive_rate    completions and runs of 20+ yards / plays
    sack_rate         sacks taken / dropbacks
    turnover_rate     interceptions and lost fumbles / plays

**Defence is not a second feed.** Every game contributes a row for both teams, so
a team's defensive line is literally its opponent's offensive line in that game.
``def_epa`` is therefore "EPA per play allowed", and ``def_sack`` is "sacks the
defence took, per opponent dropback" -- the same quantity read from the other
side. Signs are left to the fitted coefficients rather than being flipped here,
because an inverted feature that *looks* right is impossible to audit later.

**Why adjust.** NFL schedules overlap far more than college ones, so the confound
is smaller here than the 0.30 points cfb-model measures -- but it is not zero:
under the current formula a team plays six games inside its own division and
three against a single other division, and a first-place schedule is a real
thing. Measured on 2,439 out-of-sample games (2017-2025, leave-one-season-out),
adjusting is worth 0.09 points of MAE. Small, kept because it is cheap and it
adjusts the right quantity.

The solve is the same joint iteration the power ratings use:

    adj_off_i = mean over games of ( off_ij + (league_def_mean - adj_def_j) )
    adj_def_j = mean over games of ( def_ji + (league_off_mean - adj_off_i) )
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from . import teams as teams_mod
from .sources.nflverse import number

# Adjusted per-play rates. Order fixes the order everywhere downstream.
ADJUSTABLE_STATS = ("epa", "first_down", "explosive", "sack", "turnover")

ITERATIONS = 10

# Pace feeds the totals model, not the margin model, and is deliberately NOT
# opponent-adjusted: how many plays a team runs is mostly a joint property of the
# two teams, and adjusting it would double-count the opponent term the totals
# model already carries as a sum.
PACE_STATS = ("plays",)


@dataclass(frozen=True)
class GameLine:
    """One team's offensive line in one game."""

    season: int
    week: int
    game_id: str
    team: str
    opponent: str
    plays: float
    epa: float
    first_down: float
    explosive: float
    sack: float
    turnover: float

    def value(self, stat: str) -> float:
        return float(getattr(self, stat))


def _f(row: dict, key: str) -> float:
    value = number(row.get(key))
    return 0.0 if value is None else value


def game_lines(rows: list[dict], *, season_type: str = "REG") -> list[GameLine]:
    """Weekly team box scores -> per-play offensive rates, one row per team-game.

    Rows with no offensive plays are dropped rather than divided by zero. That is
    a data defect (a cancelled game, a partial publish), and a zeroed rate would
    enter the mean as a real observation of a team that never played.
    """
    out: list[GameLine] = []
    for row in rows:
        if season_type and str(row.get("season_type") or "").upper() != season_type:
            continue
        team = teams_mod.canonical(row.get("team") or "")
        opponent = teams_mod.canonical(row.get("opponent_team") or "")
        if not team or not opponent:
            continue
        attempts = _f(row, "attempts")
        carries = _f(row, "carries")
        sacks = _f(row, "sacks_suffered")
        plays = attempts + carries + sacks
        dropbacks = attempts + sacks
        if plays <= 0:
            continue
        first_downs = _f(row, "passing_first_downs") + _f(row, "rushing_first_downs")
        explosive = _f(row, "passing_20") + _f(row, "rushing_20")
        giveaways = _f(row, "passing_interceptions") + _f(row, "fumbles_lost_total")
        season = number(row.get("season")) or 0
        week = number(row.get("week")) or 0
        out.append(GameLine(
            season=int(season),
            week=int(week),
            game_id=str(row.get("game_id") or ""),
            team=team,
            opponent=opponent,
            plays=plays,
            epa=(_f(row, "passing_epa") + _f(row, "rushing_epa")) / plays,
            first_down=first_downs / plays,
            explosive=explosive / plays,
            sack=(sacks / dropbacks) if dropbacks > 0 else 0.0,
            turnover=giveaways / plays,
        ))
    return out


def _pairs(lines: list[GameLine], stat: str) -> list[tuple[str, str, float, float]]:
    """(team, opponent, own offensive value, value the team's defence allowed).

    The defensive value is looked up rather than derived: it is the opponent's
    own offensive line in the same game, which exists whenever the feed is
    complete and is skipped when it is not.
    """
    by_game: dict[str, dict[str, GameLine]] = {}
    for line in lines:
        by_game.setdefault(line.game_id, {})[line.team] = line
    out = []
    for line in lines:
        other = by_game.get(line.game_id, {}).get(line.opponent)
        if other is None:
            continue
        out.append((line.team, line.opponent, line.value(stat), other.value(stat)))
    return out


def adjust(lines: list[GameLine], stat: str, *, iterations: int = ITERATIONS
           ) -> tuple[dict[str, float], dict[str, float]]:
    """Return (adjusted offense, adjusted defense) for one stat."""
    pairs = _pairs(lines, stat)
    if not pairs:
        return {}, {}
    names = sorted({t for t, _, _, _ in pairs})
    offense_mean = statistics.fmean([o for _, _, o, _ in pairs])
    defense_mean = statistics.fmean([d for _, _, _, d in pairs])

    adj_off = dict.fromkeys(names, offense_mean)
    adj_def = dict.fromkeys(names, defense_mean)
    for _ in range(iterations):
        acc_off: dict[str, list[float]] = {t: [] for t in names}
        acc_def: dict[str, list[float]] = {t: [] for t in names}
        for team, opponent, own, allowed in pairs:
            # Credit the offense for the defence it actually faced, and vice versa.
            acc_off[team].append(own + (defense_mean - adj_def.get(opponent, defense_mean)))
            acc_def[team].append(allowed + (offense_mean - adj_off.get(opponent, offense_mean)))
        adj_off = {t: (statistics.fmean(v) if v else offense_mean) for t, v in acc_off.items()}
        adj_def = {t: (statistics.fmean(v) if v else defense_mean) for t, v in acc_def.items()}
    return adj_off, adj_def


def adjust_all(lines: list[GameLine], *, stats: tuple[str, ...] = ADJUSTABLE_STATS
               ) -> dict[str, tuple[dict[str, float], dict[str, float]]]:
    return {stat: adjust(lines, stat) for stat in stats}


def pace_means(lines: list[GameLine]) -> dict[str, dict[str, float]]:
    """Per-team mean offensive plays per game."""
    acc: dict[str, list[float]] = {}
    for line in lines:
        acc.setdefault(line.team, []).append(line.plays)
    return {team: {"plays": statistics.fmean(values)} for team, values in acc.items() if values}
