"""NFL logic matrix -- fitted on opponent-adjusted per-play efficiency.

The matrix predicts **points scored by one team in one game** from that team's
offence and the opponent's defence:

    points(A vs B) = intercept + SUM_x  g_x * ( off_x(A) + def_x(B) )  + home_field

Margin, total, and the offensive and defensive power rankings are all derived
from that one fit, which is why they reconcile exactly instead of being three
models that disagree at the edges.

Why one coefficient per family and not two
------------------------------------------
`off_epa` and `def_epa` measure the same per-play quantity in the same units --
what A's offence produced, and what B's defence allowed -- so a matchup is their
*sum*. Giving the two sides free coefficients looks more flexible and is worse:
fitted that way, `def_epa` came out **positive** (allowing more EPA per play
improved the home margin) because `epa` and `first_down` correlate at r = 0.83
and the collinear pair split the effect with opposite signs. The symmetric form
costs 0.016 points of out-of-sample MAE -- inside the noise -- and every
coefficient it produces carries the sign football says it should.

Two things live here, and they are not the same thing:

* `COEFFICIENTS` -- what the model **predicts** with.
* `OFFENSE_WEIGHTS` / `DEFENSE_WEIGHTS` -- the **interpretation**. Standardised
  importances (|coefficient| x the spread of that rate across teams), normalised
  within each group. Offence and defence share a coefficient but *not* a spread:
  team-to-team variation in offensive EPA is larger than in defensive EPA
  allowed, so the two groups weight differently, and that difference is itself
  the finding -- it is why offence separates teams more than defence does.

The coefficient specification was selected leave-one-season-out on 2,383
completed regular-season games, 2017-2025.  Its headline performance was then
re-audited with an expanding-season coefficient protocol on 1,615 games, 2020-2025:
every scored season uses coefficients fit only on earlier seasons. The feature
specification and hyperparameters were frozen at their legacy LOSO-selected 2026
values, so this is not a fully nested or prospective trial. No tested
regime or dispersion correction improved that time-forward baseline, so the
matrix keeps its measured 50/50 form instead of forcing a cosmetically balanced
favorite/underdog mix.  Reproduce with `scripts/fit_matrix.py` and
`scripts/audit_regimes.py`.

What replaced the research prior
--------------------------------
The scaffold inherited from `nfl-genesis/src/genesis/logic_matrix.py` asserted
seven offensive families with hand-assigned weights -- EPA 0.30, success rate
0.18, explosiveness 0.14, early-down pass efficiency 0.14, pressure allowed 0.12,
red zone 0.07, special teams 0.05 -- and was explicitly labelled
``CHALLENGER/UNPROMOTED``. Three of the seven are not derivable from the weekly
team box score at all, so shipping them as live weights would have meant shipping
four measured families and three decorative ones. This set drops what cannot be
measured and reports what the rest are worth.
"""

from __future__ import annotations

from dataclasses import dataclass

LINEAGE_VERSION = "2026.09-time-forward-audited-symmetric-matchup"
STATUS = "CHALLENGER/UNPROMOTED"
VALIDATION_PROTOCOL = (
    "expanding-season coefficients; frozen LOSO-selected specification; 2020-2025"
)
PERFORMANCE = {
    "games": 1615,
    "margin_mae": 10.2274,
    "market_margin_mae": 9.7644,
    "ats_rate": 0.4956,
    "underdog_side_share": 0.7232,
}
SOURCE_LINEAGE = (
    "Alphakiller1/nfl-genesis/src/genesis/logic_matrix.py (research prior, superseded)",
    "Alphakiller1/cfb-model/src/cfbmodel/matrix.py (structure)",
    "nflverse-data stats_team_week (measurement)",
)

# Prediction. Points contributed per unit of ( own offensive rate + opponent's
# defensive rate ). Signs are meaningful and all five agree with football:
# efficiency and explosive plays add points, sacks and turnovers remove them.
COEFFICIENTS: dict[str, float] = {
    "intercept": 2.875606,
    "epa": 6.988167,
    "first_down": 34.493268,
    "explosive": 54.624524,
    "sack": -21.454012,
    "turnover": -101.795473,
    "home_field": 1.739838,
}

# League-mean opponent-adjusted rate, 2017-2025. The offensive and defensive
# means of a given rate are equal by construction -- every play is produced by
# one offence against one defence -- so one number per family centres both
# indices.
LEAGUE_MEAN_FORM: dict[str, float] = {
    "epa": -0.002623,
    "first_down": 0.286567,
    "explosive": 0.059364,
    "sack": 0.066332,
    "turnover": 0.021157,
}

# Interpretation. Sums to 1.0 within each group.
OFFENSE_WEIGHTS = {
    "first_down_rate": 0.2968,
    "epa_per_play": 0.1960,
    "explosive_rate": 0.1886,
    "giveaway_rate": 0.1820,
    "sack_rate_allowed": 0.1366,
}

# Not a copy of the offence group with the labels changed. The coefficients are
# shared, the spreads are not: turnovers separate defences (0.2221) far more than
# they separate offences (0.1820), and EPA separates offences more than defences.
# Takeaway rate being a bigger share of what distinguishes a defence than of what
# distinguishes an offence is the fitted version of an old scouting claim.
DEFENSE_WEIGHTS = {
    "first_down_allowed": 0.2877,
    "takeaway_rate": 0.2221,
    "explosive_allowed": 0.1917,
    "epa_allowed": 0.1811,
    "sack_rate": 0.1174,
}

# Measured constants, mirrored from `ratings.py` so a reader of the matrix does
# not have to open a second module to see what it assumes.
HOME_FIELD_POINTS = 1.20
BLOWOUT_CAP = 42.0
RECENCY_HALFLIFE_WEEKS = 8.0
MARGIN_SD = 13.18

STATS = ("epa", "first_down", "explosive", "sack", "turnover")

GROUPS = {
    "offense": OFFENSE_WEIGHTS,
    "defense": DEFENSE_WEIGHTS,
}

# Human labels, so a breakdown reads as football rather than as variable names.
FEATURE_LABELS = {
    "epa": "EPA per play",
    "first_down": "First-down rate",
    # Kept short: the board kernel truncates a tile label with an ellipsis, and
    # "Explosive-play rate" rendered as "EXPLOSIVE-PLAY R…".
    "explosive": "Explosive rate",
    "sack": "Sack rate",
    "turnover": "Turnover rate",
}


class WeightGroupError(ValueError):
    """A weight group violates the matrix contract."""


def validate_weight_group(weights: dict[str, float], *, name: str = "group") -> None:
    """Non-negative and summing to one. Mirrors the shared genesis core."""
    if not weights:
        raise WeightGroupError(f"{name}: empty weight group")
    negative = sorted(k for k, v in weights.items() if v < 0)
    if negative:
        raise WeightGroupError(f"{name}: negative weights {negative}")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise WeightGroupError(f"{name}: must sum to 1.0, got {total!r}")


for _name, _group in GROUPS.items():
    validate_weight_group(_group, name=_name)


@dataclass(frozen=True)
class TeamForm:
    """Opponent-adjusted per-play form for one team.

    Every field is optional because a team can legitimately have no form -- a
    feed that has not published, or a franchise with no prior season. A missing
    value must never silently become zero, which on a centred rate would read as
    exactly league average.
    """

    off_epa: float | None = None
    off_first_down: float | None = None
    off_explosive: float | None = None
    off_sack: float | None = None
    off_turnover: float | None = None
    def_epa: float | None = None
    def_first_down: float | None = None
    def_explosive: float | None = None
    def_sack: float | None = None
    def_turnover: float | None = None
    # Pace. Carried for display and for downstream consumers, but NOT a model
    # feature: adding plays-per-game to the points model made the total 0.014
    # WORSE out of sample. NFL tempo barely varies, which is the opposite of the
    # college case, where cfb-model's totals model needs both drives and plays.
    plays: float | None = None

    FIELDS = ("off_epa", "off_first_down", "off_explosive", "off_sack", "off_turnover",
              "def_epa", "def_first_down", "def_explosive", "def_sack", "def_turnover")

    def complete(self) -> bool:
        return all(getattr(self, f) is not None for f in self.FIELDS)


def points(offense: TeamForm, defence: TeamForm, *, home: bool) -> float | None:
    """Points one side is projected to score, or None if either form is missing.

    `offense` supplies the scoring side's ``off_*`` rates; `defence` supplies the
    opponent's ``def_*`` rates.
    """
    if not (offense.complete() and defence.complete()):
        return None
    value = COEFFICIENTS["intercept"]
    for stat in STATS:
        value += COEFFICIENTS[stat] * (getattr(offense, f"off_{stat}")
                                       + getattr(defence, f"def_{stat}"))
    if home:
        value += COEFFICIENTS["home_field"]
    return value


def margin_points(home: TeamForm, away: TeamForm, *, neutral: bool = False) -> float | None:
    """Efficiency-only expected home margin.

    Equal to (offence + defence) index of the home team minus the away team's,
    plus home field -- the intercept and the league-mean terms cancel out of a
    difference, which is why this reconciles with the two indices exactly.
    """
    home_points = points(home, away, home=not neutral)
    away_points = points(away, home, home=False)
    if home_points is None or away_points is None:
        return None
    return home_points - away_points


def offense_index(form: TeamForm) -> float | None:
    """Points per game this offence generates above an average one.

    On the same points scale as the overall rating rather than an invented 0-100
    index, so an offence rated +3.1 and a defence rated +1.4 add to a +4.5 team.
    """
    if any(getattr(form, f"off_{s}") is None for s in STATS):
        return None
    return sum(COEFFICIENTS[s] * (getattr(form, f"off_{s}") - LEAGUE_MEAN_FORM[s])
               for s in STATS)


def defense_index(form: TeamForm) -> float | None:
    """Points per game this defence prevents relative to an average one.

    `def_*` rates are what opponents produced, so a good defence has low def_epa
    and high def_sack. The sum is negated once, here, so that positive always
    means better -- the one place a sign error would be invisible on a ranking
    table, since a plausible-looking order would still render.
    """
    if any(getattr(form, f"def_{s}") is None for s in STATS):
        return None
    return -sum(COEFFICIENTS[s] * (getattr(form, f"def_{s}") - LEAGUE_MEAN_FORM[s])
                for s in STATS)


def efficiency_rating(form: TeamForm) -> float | None:
    """Offence plus defence: this team's total efficiency edge, in points."""
    offense, defence = offense_index(form), defense_index(form)
    if offense is None or defence is None:
        return None
    return offense + defence


def margin_contributions(home: TeamForm, away: TeamForm) -> dict[str, float] | None:
    """Each family's contribution to the efficiency margin, in points.

    Because the matchup model is linear and symmetric, the margin decomposes
    exactly:

        efficiency_margin = home_field + SUM_x g_x * [ (off_x(H) + def_x(A))
                                                     - (off_x(A) + def_x(H)) ]

    so these values plus `COEFFICIENTS["home_field"]` reconcile to
    `margin_points` with no residual term. That exactness is the point: a
    breakdown whose parts do not add up to the number above it is worse than no
    breakdown, because it invites the reader to trust an accounting that is
    quietly wrong.

    Positive favours the home team.
    """
    if not (home.complete() and away.complete()):
        return None
    return {
        stat: COEFFICIENTS[stat] * (
            (getattr(home, f"off_{stat}") + getattr(away, f"def_{stat}"))
            - (getattr(away, f"off_{stat}") + getattr(home, f"def_{stat}"))
        )
        for stat in STATS
    }
