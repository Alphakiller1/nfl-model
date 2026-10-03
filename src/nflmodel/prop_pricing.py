"""Price a player prop from the projection matrix, calibrated against the line.

Each quote is priced from the projection itself, not from a market base
rate:

1. ``raw`` = P(stat > line) under the outcome distribution fitted on the
   projection layer's own errors (`prop_distributions`), centred on the
   matrix projection.
2. ``p_over`` = 0.5 + s * (raw - c). The raw distribution is honest about the
   player's spread of outcomes, but it ignores what the line knows. Against
   DraftKings closing lines a raw 70% hit 57%, so it is shrunk toward even
   by a factor s fitted leave-one-week-out. c is the raw value that
   corresponds to an even line.
3. ``fair`` = line + w * (projection - line): how far toward the projection
   the number should move, by market family (reports/PROPS_MODEL.md, C1).

Evidence (EVIDENCE): matrix projections rebuilt point-in-time for 2026 weeks
1-3 and priced against 1,920 DraftKings closing lines. Leave-one-week-out,
the plays priced at >= 53% went 450-388 (53.7%; -110 break-even 52.4%) with
a held-out Brier of 0.2499 against a coin's 0.2500. Ranking players within
each week and market, top third over and bottom third under, went 688-568
(54.8%), with no fitting beyond the projection. The fair-number blend weight
on these projections is 0.59 for receiving (0.37 on the previous version):
the line wants to move further toward them. ``research_only`` holds until
the held-out record clears the -110 break-even on enough plays with a
one-sided 90% lower bound above a coin.
"""

from __future__ import annotations

import math

from . import prop_distributions

FAMILY = {
    "receiving_yards": "receiving", "receptions": "receiving", "targets": "receiving",
    "rushing_yards": "rushing", "carries": "rushing", "rush_attempts": "rushing",
    "passing_yards": "passing", "passing_tds": "passing", "completions": "passing",
    "pass_attempts": "passing", "interceptions": "passing",
}
BLEND_WEIGHT = {"passing": 0.422, "receiving": 0.586, "rushing": 0.167}
# scripts/fit_prop_pricing.py --rows <2026 weeks 1-3 point-in-time replay lines>.
CALIBRATION = {"s": 0.30, "c": 0.505}
EVIDENCE = {
    "source": "2026 weeks 1-3, point-in-time matrix replay vs DraftKings closing lines",
    "lines": 1920, "plays": 838, "wins": 450, "hit_rate": 0.537,
    "ranked_wins": 688, "ranked_plays": 1256,
}
MIN_PROBABILITY = 0.53
BREAK_EVEN = 0.524      # -110 both ways
MIN_EVIDENCE_PLAYS = 300


def _lower_bound(wins: int, plays: int) -> float:
    if not plays:
        return 0.0
    rate = wins / plays
    return rate - 1.2816 * math.sqrt(rate * (1 - rate) / plays)


def research_only() -> bool:
    """True until the held-out record clears break-even on enough plays and
    its one-sided 90% lower bound beats a coin."""
    return not (EVIDENCE["plays"] >= MIN_EVIDENCE_PLAYS
                and EVIDENCE["hit_rate"] > BREAK_EVEN
                and _lower_bound(EVIDENCE["wins"], EVIDENCE["plays"]) > 0.5)


def calibrate(raw: float) -> float:
    return min(max(0.5 + CALIBRATION["s"] * (raw - CALIBRATION["c"]), 0.01), 0.99)


def price(metric: str, projection: float | None, line: float | None) -> dict | None:
    """{'family', 'fair', 'raw', 'p_over'} for one quote, or None if unpriceable."""
    family = FAMILY.get(metric)
    if family is None or projection is None or line is None or projection <= 0:
        return None
    dist = prop_distributions.distribution(metric, projection)
    raw = prop_distributions.over_probability(dist, float(line)) if dist else None
    if raw is None:
        return None
    return {
        "family": family,
        "fair": round(float(line) + BLEND_WEIGHT[family] * (float(projection) - float(line)), 2),
        "raw": round(raw, 4),
        "p_over": round(calibrate(raw), 4),
    }
