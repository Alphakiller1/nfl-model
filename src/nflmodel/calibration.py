"""Every published probability, priced on the skill its market has shown.

A model number is not the truth: the closing line is better informed on every
market this model has measured. So a pick's probability is the share of the
model-minus-market gap that has historically shown up in results (the blend
weight w), pushed through the outcome's spread:

    P(model side) = Phi(w * |gap| / sigma)

and a pick publishes only when that clears MIN_PUBLISH_PROBABILITY. A market
whose measured weight is ~0 therefore publishes nothing, however large the
disagreement looks: that is the point. Before this module, spread and total
picks were priced with w = 1 (GB @ TB, week 4 2026: "Over 38.5, 76%" on a
9.3-point gap the model has never been able to cash).

Evidence is time-forward: coefficients fitted on earlier seasons, scored on the
next (research/props_model/model_audit.py, qb_out_totals.py). Refit after
each season; `scripts/audit_consistency.py` fails a deploy that publishes a
pick priced above these weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MarketSkill:
    weight: float        # share of (model - market) that shows up in the result
    sigma: float         # SD of the outcome around the market
    evidence: str


SKILL = {
    "spread": MarketSkill(
        weight=0.0, sigma=13.2,
        evidence=("2020-2025 time-forward, QB availability applied: blend weight -0.035 "
                  "(seasons -0.50 to +0.29); ATS 49.1% at 2+ point gaps (792), 50.3% at 4+ "
                  "(298), 51.6% at 6+ (95); margin MAE model 10.15 vs market 9.76.")),
    "total": MarketSkill(
        weight=0.02, sigma=12.9,
        evidence=("2021-2025 time-forward, QB availability applied: blend weight 0.017 "
                  "(seasons -0.30 to +0.23); model side 233-222 at 3+ point gaps, 52-52 "
                  "at 6+.")),
}
MIN_PUBLISH_PROBABILITY = 0.53


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def probability(market: str, gap: float) -> float:
    """Probability the model's side wins, given its measured skill on `market`."""
    skill = SKILL[market]
    return _phi(skill.weight * abs(float(gap)) / skill.sigma)


def publishable(market: str, gap: float) -> bool:
    return probability(market, gap) >= MIN_PUBLISH_PROBABILITY


def ceiling(market: str, gap: float, tolerance: float = 1e-3) -> float:
    """The highest probability a published pick on `market` may carry."""
    return probability(market, gap) + tolerance
