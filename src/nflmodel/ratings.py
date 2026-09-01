"""Opponent-adjusted power ratings.

A rating is points relative to an average team on a neutral field, so the
projected neutral margin between two teams is the difference of their ratings and
home field adds `HOME_FIELD_POINTS` to the host.

    rating_i = weighted mean over games of ( own capped margin + opponent rating )

Home field is removed from each margin before rating, so a team is not credited
for a soft home schedule, and ratings are re-centred on the league mean after
every pass.

Every constant below was selected by walk-forward test on the 2,383 completed
regular-season games of 2017-2025 -- the model is refitted without the season it
is scored on, and the parameter is chosen on out-of-sample MAE of the published
(blended) margin. `reports/BASELINE_2016_2025.md` carries the sweeps.

Two of those sweeps came back flat, and saying so is more useful than presenting
a tuned number as if it were earned.

Capping barely matters here, unlike in college
----------------------------------------------
cfb-model caps margins at 32 and buys half a point of MAE, because 36% of FBS
games are decided by 28 or more. The NFL figure is **7.0%**. The same sweep run on
NFL data moves MAE by 0.004 points across caps from 17 to none -- nothing. The cap
is kept at 42 (the sweep's argmin) purely to bound what a single freak Thursday
result can do to a September rating, and it should not be described as a
contributor. Copying college's 32 unexamined would have looked more decisive and
meant less.

Home field: measured 2.06, published 1.20, and the gap is not an error
----------------------------------------------------------------------
The raw mean home margin is +1.90 across 2014-2019 and +2.06 across 2021-2025 --
it did not decline the way the post-2020 commentary suggested. But the constant
that minimises out-of-sample error *here* is 1.20, and the reason is structural
rather than empirical: the published margin is a 50/50 blend of this rating path
and the efficiency path (see `totals.py`), and the efficiency model carries its
own fitted home-field term of 1.74. The blend therefore applies
0.5 x 1.20 + 0.5 x 1.74 = **1.47** points of home field in total. Setting this
constant to the measured 2.06 would double-count with the other half of the blend
and overstate home advantage.

Recency half-life was swept from 4 weeks to no decay at all and moved MAE by
0.007 points. It is set to 8 -- an interior, near-optimal value -- and the
parameter should be read as unidentified rather than tuned.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from . import teams as teams_mod

# Selected on out-of-sample MAE. See the module docstring and reports/.
BLOWOUT_CAP = 42.0
HOME_FIELD_POINTS = 1.20
RECENCY_HALFLIFE_WEEKS = 8.0
SHRINK = 1.0
ITERATIONS = 15

# Scale for turning a projected margin into a win probability. This is the
# RESIDUAL standard deviation of the blended forecast (13.32), not the raw spread
# of NFL margins (14.32). P(home wins) is P(actual > 0 | forecast), so the
# relevant dispersion is the model's error, not the league's variance -- using
# the raw 14.32 would push every probability toward 50% by claiming more
# uncertainty than the forecast actually has.
MARGIN_SD = 13.32

GAMES_PER_SEASON = 17

# Sentinel for "use the module constant". A plain default argument binds once at
# import, so a sweep that reassigns BLOWOUT_CAP or RECENCY_HALFLIFE_WEEKS and
# then calls build() would silently keep the original value and report a flat
# curve -- which is exactly how the first version of the blend sweep in
# scripts/fit_matrix.py measured nothing. `halflife=None` is a real setting (no
# recency decay), so absence needs its own marker rather than None.
_DEFAULT = object()


def cap_margin(margin: float, cap=_DEFAULT) -> float:
    """Smoothly compress a margin toward `cap`, preserving order.

    `cap * tanh(margin / cap)` rather than a hard clip, so a 45-point win still
    outranks a 30-point one instead of every blowout collapsing to one value.
    """
    cap = BLOWOUT_CAP if cap is _DEFAULT else cap
    if cap <= 0:
        return margin
    return cap * math.tanh(margin / cap)


@dataclass(frozen=True)
class Game:
    season: int
    week: int
    home: str
    away: str
    home_points: float
    away_points: float
    neutral: bool = False

    @property
    def margin(self) -> float:
        return float(self.home_points - self.away_points)


def from_rows(rows: list[dict], *, completed_only: bool = True) -> list[Game]:
    """nflverse schedule rows -> rateable games, with relocations folded forward."""
    out: list[Game] = []
    for row in rows:
        if str(row.get("game_type") or "").upper() != "REG":
            continue
        home_points, away_points = row.get("home_score"), row.get("away_score")
        if completed_only and (home_points is None or away_points is None):
            continue
        out.append(Game(
            season=int(row.get("season") or 0),
            week=int(row.get("week") or 0),
            home=teams_mod.canonical(row.get("home_team") or ""),
            away=teams_mod.canonical(row.get("away_team") or ""),
            home_points=float(home_points or 0.0),
            away_points=float(away_points or 0.0),
            neutral=str(row.get("location") or "Home").lower() != "home",
        ))
    return out


def build(
    games: list[Game],
    *,
    cap=_DEFAULT,
    home_field=_DEFAULT,
    halflife=_DEFAULT,
    iterations: int = ITERATIONS,
    season_weights: dict[int, float] | None = None,
) -> dict[str, float]:
    """Solve the rating system, centred on the league mean.

    `season_weights` lets a caller weight whole seasons -- the preseason prior
    uses it to discount older years. Recency inside a season is handled by
    `halflife`, measured in weeks from the most recent game in the sample.
    """
    cap = BLOWOUT_CAP if cap is _DEFAULT else cap
    home_field = HOME_FIELD_POINTS if home_field is _DEFAULT else home_field
    halflife = RECENCY_HALFLIFE_WEEKS if halflife is _DEFAULT else halflife
    if not games:
        return {}
    latest_season = max(g.season for g in games)
    latest_week = max(g.week for g in games if g.season == latest_season)
    observations: list[tuple[str, str, float, float]] = []
    for g in games:
        adjustment = 0.0 if g.neutral else home_field
        margin = cap_margin(g.margin - adjustment, cap)
        # Distance in weeks from the end of the sample, counting whole seasons at
        # a nominal 22 weeks so a January game is not treated as a decade old.
        age = (latest_season - g.season) * 22 + (latest_week - g.week)
        weight = 1.0 if halflife is None else 0.5 ** (age / halflife)
        if season_weights is not None:
            weight *= season_weights.get(g.season, 0.0)
        if weight <= 0.0:
            continue
        observations.append((g.home, g.away, margin, weight))
        observations.append((g.away, g.home, -margin, weight))

    names = sorted({team for team, _, _, _ in observations})
    ratings = dict.fromkeys(names, 0.0)
    for _ in range(iterations):
        numerator = dict.fromkeys(names, 0.0)
        denominator = dict.fromkeys(names, 0.0)
        for team, opponent, margin, weight in observations:
            numerator[team] += weight * (margin + ratings.get(opponent, 0.0))
            denominator[team] += weight
        updated = {t: (numerator[t] / denominator[t] if denominator[t] else 0.0) for t in names}
        centre = statistics.fmean(updated.values()) if updated else 0.0
        ratings = {t: v - centre for t, v in updated.items()}
    return {t: v * SHRINK for t, v in ratings.items()}


def projected_margin(ratings: dict[str, float], home: str, away: str, *,
                     neutral: bool = False, home_field=_DEFAULT) -> float | None:
    """Expected home margin, or None when either team is unrated."""
    home_field = HOME_FIELD_POINTS if home_field is _DEFAULT else home_field
    home, away = teams_mod.canonical(home), teams_mod.canonical(away)
    if home not in ratings or away not in ratings:
        return None
    return ratings[home] - ratings[away] + (0.0 if neutral else home_field)


def win_probability(margin: float, *, sd=_DEFAULT) -> float:
    """Normal CDF of the projected margin. Symmetric, never exactly 0 or 1."""
    sd = MARGIN_SD if sd is _DEFAULT else sd
    return 0.5 * (1.0 + math.erf(float(margin) / (sd * math.sqrt(2.0))))


def rank_table(ratings: dict[str, float]) -> list[tuple[int, str, float]]:
    """(rank, team, rating), strongest first."""
    ordered = sorted(ratings.items(), key=lambda kv: -kv[1])
    return [(i, team, value) for i, (team, value) in enumerate(ordered, start=1)]
