"""Authority gate — what a forecast is *allowed* to be used for.

A probability and a permission are different things, and conflating them is how an
unpromoted model becomes a bet. This module keeps them separate: `forecast.py`
produces numbers, and nothing downstream may act on them without an `Authority`
that says the evidence supports it.

Current state, from `nfl-genesis` `reports/EMPIRICAL_BASELINE_2016_2025.md`:

    market                    log loss 0.608393
    market-anchored stacker   log loss 0.608567   (-0.000173)
    selected lam              0.135, 0.045, then 0.000 for 2021-2025

Five consecutive folds selecting lam = 0 means the structural component carries no
incremental information over a paired no-vig closing line. The model therefore
*matches* the market and does not beat it. Matching a benchmark is not edge, so the
authority is RESEARCH_ONLY and `may_bet` is False.

This is deliberately hard to override. `promote()` requires every gate in
`nfl-genesis` `registry/production_requirements.yaml` to be satisfied and passed in
explicitly; there is no boolean flag that flips it on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Level(str, Enum):
    """Ordered from least to most permission."""

    RESEARCH_ONLY = "RESEARCH_ONLY"   # numbers only; never sized, never staked
    SHADOW = "SHADOW"                 # logged as if traded, no capital
    PROMOTED = "PROMOTED"             # may emit BET


class Action(str, Enum):
    AVOID = "AVOID"        # no usable price, or a failed gate
    MONITOR = "MONITOR"    # priced and modelled, but not promoted
    REVIEW = "REVIEW"      # edge implausible enough to suspect the inputs
    BET = "BET"            # promoted only


# From nfl-genesis registry/production_requirements.yaml. Listed explicitly so a
# reader can see exactly what is unmet rather than trusting a single flag.
REQUIRED_GATES: tuple[str, ...] = (
    "historical_seasons_at_least_6",
    "out_of_sample_games_at_least_1500",
    "out_of_sample_bets_at_least_300",
    "timestamped_quote_coverage_at_least_0.98",
    "model_log_loss_below_paired_no_vig_market",
    "model_brier_below_paired_no_vig_market",
    "roi_lower_confidence_bound_above_zero_after_cost",
    "probability_space_clv_above_zero",
    "deflated_sharpe_probability_at_least_0.95",
    "probability_backtest_overfit_at_most_0.20",
    "calibration_ece_at_most_0.025",
    "shadow_days_at_least_90",
)

# Gates the published evidence currently satisfies. Deliberately short.
SATISFIED_GATES: tuple[str, ...] = (
    "historical_seasons_at_least_6",            # 2016-2025
    "out_of_sample_games_at_least_1500",        # 1,865
    "calibration_ece_at_most_0.025",            # market 0.0221, anchored 0.0242
)

# An "edge" larger than this against a liquid closing line is far more likely to be
# a stale quote or a mapping error than a real disagreement. Two scales, because
# the two forecasts speak different units: probability for the moneyline contract,
# points for the spread board.
IMPLAUSIBLE_EDGE = 0.15
IMPLAUSIBLE_EDGE_POINTS = 14.0


@dataclass(frozen=True)
class Authority:
    level: Level
    unmet_gates: tuple[str, ...] = field(default_factory=tuple)
    evidence: str = ""

    @property
    def may_bet(self) -> bool:
        return self.level is Level.PROMOTED and not self.unmet_gates

    def action_for(self, edge: float | None, has_price: bool, *,
                   modelled: bool = False,
                   implausible: float = IMPLAUSIBLE_EDGE) -> Action:
        """Map a modelled edge to what may actually be done with it.

        `modelled` separates two states that a bare `edge is None` cannot.
        A game with no usable price is AVOID. A game that *is* priced and *was*
        modelled, but whose difference from the price is not publishable as an
        edge, is MONITOR -- which is precisely the definition above: priced and
        modelled, but not promoted.

        The distinction is not cosmetic. The spread board withholds the edge on
        every game by design (the model does not beat the closing line), so
        without it a fully working board would report AVOID on all sixteen cards
        and read as a feed outage rather than as an honest research-only state.
        """
        if not has_price:
            return Action.AVOID
        if edge is None:
            return Action.MONITOR if modelled else Action.AVOID
        if abs(edge) > implausible:
            # Loud on purpose: against an efficient closing line this is a bug
            # signal, not an opportunity.
            return Action.REVIEW
        return Action.BET if self.may_bet else Action.MONITOR


def current() -> Authority:
    """The authority the published NFL evidence actually supports today."""
    unmet = tuple(g for g in REQUIRED_GATES if g not in SATISFIED_GATES)
    return Authority(
        level=Level.RESEARCH_ONLY,
        unmet_gates=unmet,
        evidence=(
            "nfl-genesis reports/EMPIRICAL_BASELINE_2016_2025.md — anchored stacker "
            "-0.000173 log loss vs paired closing market; selected lam = 0.000 in all "
            "five folds 2021-2025, i.e. no incremental signal over the closing line"
        ),
    )


def promote(satisfied: set[str]) -> Authority:
    """Promote only when every required gate is explicitly satisfied.

    Takes the satisfied set as an argument rather than reading a config flag, so
    promotion is always an auditable claim about evidence.
    """
    unmet = tuple(g for g in REQUIRED_GATES if g not in satisfied)
    if unmet:
        return Authority(Level.RESEARCH_ONLY, unmet, "promotion refused")
    return Authority(Level.PROMOTED, (), "all production gates satisfied")
