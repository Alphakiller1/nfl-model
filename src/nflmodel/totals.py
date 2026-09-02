"""Projected totals and scorelines, and the two shrinkages they depend on.

The matchup model in `matrix.py` already produces both teams' points, so a total
is their sum and a margin is their difference. What lives here is the calibration
that turns those raw sums into numbers worth publishing -- and both corrections
are shrinkages, because both raw estimates are too confident.

**The margin blend.** Two estimates of the same quantity exist: the rating gap
(`ratings.projected_margin`) and the efficiency margin. Neither dominates.
Selected leave-one-season-out on 2,383 games; rechecked with a stricter
expanding-season simulation on 1,615 games:

    ratings only            MAE 10.3374
    efficiency only         MAE 10.3256
    50/50 blend             MAE 10.2274   <- retained
    market                  MAE  9.7644

The blend is worth about 0.11 points over either component, and the curve is flat
between w = 0.45 and w = 0.55, so 0.50 is the robust middle rather than a peak
found by searching. It remains 0.46 points worse than the closing line.

**The total shrink.** The raw model total is over-dispersed: regressing actual
totals on it gives a slope near 0.63, meaning it spreads its predictions wider
than reality justifies. Shrinking toward the league mean fixes that:

    prior-only league mean  MAE 10.9702
    shrunk, lam = 0.70      MAE 10.6598   <- retained
    market                  MAE 10.2833

Read those numbers honestly. The shrunk model beats a past-only constant by 0.31 points and
loses to the market by 0.38. A projected total is a centre of mass with a
13.48-point time-forward residual SD against an actual-total SD of 13.84 -- it explains very
little of the variance, and a scoreline printed to the point looks far more
precise than it is. That is why the dashboard shows the market total beside it.

**Projected scores** are algebra on the two projections:

    home = (total + margin) / 2
    away = (total - margin) / 2

They inherit the error of *both*, so read a scoreline as a centre of mass, never
as a prediction of the actual score.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import matrix

# Share of the published margin taken from the efficiency model; the rest comes
# from the opponent-adjusted rating gap. See the module docstring.
EFFICIENCY_WEIGHT = 0.50

# Shrink of the raw model total toward the league mean.
TOTAL_SHRINK = 0.70

# Measured 2017-2025.
LEAGUE_MEAN_TOTAL = 45.59
TOTAL_SD = 13.48
ACTUAL_TOTAL_SD = 13.84


@dataclass(frozen=True)
class Projection:
    total: float | None
    margin: float | None
    home_score: float | None
    away_score: float | None
    modelled: bool          # False when the total fell back to the league mean
    # The efficiency component on its own, carried out rather than recomputed by
    # the caller. Two call sites deriving the same quantity from the same inputs
    # is one refactor away from them disagreeing.
    efficiency_margin: float | None = None
    rating_margin: float | None = None


def blend_margin(rating_margin: float | None,
                 efficiency_margin: float | None,
                 *, weight: float = EFFICIENCY_WEIGHT) -> float | None:
    """Combine the two margin estimates, using whichever exist."""
    if rating_margin is None:
        return efficiency_margin
    if efficiency_margin is None:
        return rating_margin
    return weight * efficiency_margin + (1.0 - weight) * rating_margin


def shrink_total(raw: float | None, *, shrink: float = TOTAL_SHRINK) -> float | None:
    if raw is None:
        return None
    return LEAGUE_MEAN_TOTAL + shrink * (raw - LEAGUE_MEAN_TOTAL)


def project(home: matrix.TeamForm | None, away: matrix.TeamForm | None, *,
            rating_margin: float | None = None, neutral: bool = False) -> Projection:
    """Turn two forms plus a rating gap into a margin, a total and a scoreline.

    With no usable form the total falls back to the league mean rather than
    refusing to produce a scoreline -- a centred guess is more useful than a
    blank, and `modelled` marks which one the caller got.
    """
    home_points = away_points = None
    if home is not None and away is not None:
        home_points = matrix.points(home, away, home=not neutral)
        away_points = matrix.points(away, home, home=False)

    modelled = home_points is not None and away_points is not None
    efficiency_margin = (home_points - away_points) if modelled else None
    margin = blend_margin(rating_margin, efficiency_margin)
    total = shrink_total(home_points + away_points) if modelled else LEAGUE_MEAN_TOTAL

    if margin is None:
        return Projection(total if modelled else None, None, None, None, modelled,
                          efficiency_margin, rating_margin)

    home_score = (total + margin) / 2.0
    away_score = (total - margin) / 2.0
    # A large projected margin against a modest total can drive one side below
    # zero, which is not a scoreline. Clamp, and keep the total intact.
    if away_score < 0:
        away_score, home_score = 0.0, total
    elif home_score < 0:
        home_score, away_score = 0.0, total
    return Projection(total, margin, home_score, away_score, modelled,
                      efficiency_margin, rating_margin)
