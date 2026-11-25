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

Recency: a design inconsistency that costs nothing
--------------------------------------------------
This adjustment used an unweighted mean over the whole sample while
`ratings.build` decayed at an 8-week halflife, so the two halves of the
published 50/50 margin blend disagreed about what "current" meant. Fixing the
inconsistency is free and worth almost exactly nothing, which is the finding.

Swept in `scripts/sweep_margin.py`, strictly expanding, 1,615 games 2020-2025:

    halflife   none      4       6       8      12      16      24
    margin MAE 10.2274 10.2283 10.2270 10.2266 10.2265 10.2267 10.2268
    ATS %      49.56   49.81   49.68   49.81   49.75   49.87   49.56

The whole curve spans 0.0018 points of MAE and the ATS differences are a
seventh of their own standard error. `HALFLIFE_WEEKS = 8` is set to agree with
`ratings.RECENCY_HALFLIFE_WEEKS` rather than because it was earned: read the
parameter as unidentified, exactly as `ratings.py` says of its own.

The first version of that sweep measured a perfectly flat curve for a different
reason -- see `_DEFAULT` below.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from . import teams as teams_mod
from .sources.nflverse import number

# Adjusted per-play rates. Order fixes the order everywhere downstream.
ADJUSTABLE_STATS = ("epa", "first_down", "explosive", "sack", "turnover")

ITERATIONS = 10

# Recency decay for the opponent adjustment, in weeks. `None` is no decay.
# Swept in the expanding-season harness; see the module docstring.
HALFLIFE_WEEKS: float | None = 8.0

# Sentinel for "use the module constant", mirroring `ratings.py`. A plain
# default argument binds HALFLIFE_WEEKS once at import, so a sweep that
# reassigns it and then calls adjust() keeps the original value and reports a
# perfectly flat curve. `ratings.py` carries the same warning because the same
# mistake was made there first; it was made here too, and the sweep returned
# identical MAE to four decimals for every halflife from 4 to 24.
# `halflife=None` is a real setting (no decay), so absence needs its own marker.
_DEFAULT = object()

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


def _pairs(lines: list[GameLine], stat: str, *, halflife: float | None = None
           ) -> list[tuple[str, str, float, float, float]]:
    """(team, opponent, own offensive value, value the defence allowed, weight).

    The defensive value is looked up rather than derived: it is the opponent's
    own offensive line in the same game, which exists whenever the feed is
    complete and is skipped when it is not.
    """
    by_game: dict[str, dict[str, GameLine]] = {}
    for line in lines:
        by_game.setdefault(line.game_id, {})[line.team] = line
    weights = _weights(lines, halflife)
    out = []
    for line in lines:
        other = by_game.get(line.game_id, {}).get(line.opponent)
        if other is None:
            continue
        weight = weights.get((line.game_id, line.season, line.week), 1.0)
        if weight <= 0.0:
            continue
        out.append((line.team, line.opponent, line.value(stat), other.value(stat), weight))
    return out


def _weights(lines: list[GameLine], halflife: float | None) -> dict[tuple[str, int, int], float]:
    """Recency weight per team-game, halving every `halflife` weeks."""
    if halflife is None:
        return {}
    latest_season = max(line.season for line in lines)
    latest_week = max(line.week for line in lines if line.season == latest_season)
    out: dict[tuple[str, int, int], float] = {}
    for line in lines:
        # Whole seasons count as a nominal 22 weeks, matching `ratings.build`,
        # so a January game is not treated as a decade old.
        age = (latest_season - line.season) * 22 + (latest_week - line.week)
        out[(line.game_id, line.season, line.week)] = 0.5 ** (age / halflife)
    return out


def adjust(lines: list[GameLine], stat: str, *, iterations: int = ITERATIONS,
           halflife=_DEFAULT) -> tuple[dict[str, float], dict[str, float]]:
    """Return (adjusted offense, adjusted defense) for one stat.

    `halflife` decays older games, in weeks. See the module docstring: this used
    to be an unweighted mean over the whole sample while `ratings.build` decayed
    at an 8-week halflife, so the two halves of the published 50/50 blend
    disagreed about what "current" meant.
    """
    halflife = HALFLIFE_WEEKS if halflife is _DEFAULT else halflife
    pairs = _pairs(lines, stat, halflife=halflife)
    if not pairs:
        return {}, {}
    names = sorted({t for t, _, _, _, _ in pairs})
    total_weight = sum(w for _, _, _, _, w in pairs) or 1.0
    offense_mean = sum(o * w for _, _, o, _, w in pairs) / total_weight
    defense_mean = sum(d * w for _, _, _, d, w in pairs) / total_weight

    adj_off = dict.fromkeys(names, offense_mean)
    adj_def = dict.fromkeys(names, defense_mean)
    for _ in range(iterations):
        acc_off: dict[str, list[tuple[float, float]]] = {t: [] for t in names}
        acc_def: dict[str, list[tuple[float, float]]] = {t: [] for t in names}
        for team, opponent, own, allowed, weight in pairs:
            # Credit the offense for the defence it actually faced, and vice versa.
            acc_off[team].append(
                (own + (defense_mean - adj_def.get(opponent, defense_mean)), weight))
            acc_def[team].append(
                (allowed + (offense_mean - adj_off.get(opponent, offense_mean)), weight))
        adj_off = {t: _weighted_mean(v, offense_mean) for t, v in acc_off.items()}
        adj_def = {t: _weighted_mean(v, defense_mean) for t, v in acc_def.items()}
    return adj_off, adj_def


def _weighted_mean(pairs: list[tuple[float, float]], fallback: float) -> float:
    total = sum(w for _, w in pairs)
    if total <= 0:
        return fallback
    return sum(v * w for v, w in pairs) / total


def adjust_all(lines: list[GameLine], *, stats: tuple[str, ...] = ADJUSTABLE_STATS,
               halflife=_DEFAULT
               ) -> dict[str, tuple[dict[str, float], dict[str, float]]]:
    return {stat: adjust(lines, stat, halflife=halflife) for stat in stats}


def pace_means(lines: list[GameLine]) -> dict[str, dict[str, float]]:
    """Per-team mean offensive plays per game."""
    acc: dict[str, list[float]] = {}
    for line in lines:
        acc.setdefault(line.team, []).append(line.plays)
    return {team: {"plays": statistics.fmean(values)} for team, values in acc.items() if values}
