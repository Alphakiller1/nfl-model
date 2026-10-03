"""Price a player prop from the DraftKings line plus a shrunk model gap.

Measured on 2026 weeks 1-3 (1,949 DraftKings closing lines, see
reports/PROPS_MODEL.md part C), the line beats the raw projection on every
market. The projection still carries information the line lacks, but only a
fraction of its disagreement is real: the error-minimising blend is
``line + w * (projection - line)`` with w of about 0.3-0.4 for receiving
markets. Reading the raw projection through a skewed outcome distribution
(the old path) produced probabilities with a worse Brier score than a coin.

So a quote is priced in two numbers:

* ``fair``: the line moved toward the projection by the fitted weight w.
* ``p_over``: a logistic on the gap scaled by the line's size,
  ``sigmoid(a + b * (projection - line) / sqrt(line))``, fitted per market
  family. The intercept carries the families' measured over/under lean.

Coefficients are refit by ``scripts/fit_prop_pricing.py`` (leave-one-week-out)
and pasted here. ``EVIDENCE`` holds that script's held-out record of the
probability >= 55% plays. Until it clears the -110 break-even (52.4%) on
enough plays, no prop pick is published; lines and prices are still recorded
in the ledger and graded, so the evidence accumulates.
"""

from __future__ import annotations

import math

FAMILY = {
    "receiving_yards": "receiving", "receptions": "receiving", "targets": "receiving",
    "rushing_yards": "rushing", "carries": "rushing", "rush_attempts": "rushing",
    "passing_yards": "passing", "passing_tds": "passing", "completions": "passing",
    "pass_attempts": "passing", "interceptions": "passing",
}
# family -> blend weight w, logistic intercept a, logistic slope b.
# Fitted 2026-10-03 on 2,003 lines (2026 weeks 1-4) by scripts/fit_prop_pricing.py.
# The intercept is held at 0. Fitted freely it carried the season's over/under
# base rate (receiving overs hit 46%) and priced 269 receiving unders and 0
# overs in week 1: a constant, not a read on any player. Through 50%, the
# receiving and rushing slopes are ~0, so those families price no plays.
COEFFICIENTS = {
    "passing": {"w": 0.374, "a": 0.0, "b": 0.239},
    "receiving": {"w": 0.374, "a": 0.0, "b": -0.008},
    "rushing": {"w": 0.203, "a": 0.0, "b": 0.016},
}
# Held-out (leave-one-week-out) record of plays priced >= MIN_PROBABILITY.
EVIDENCE = {"weeks": ["2026-1", "2026-2", "2026-3", "2026-4"], "plays": 69, "wins": 32,
            "hit_rate": 0.464}
MIN_PROBABILITY = 0.55
BREAK_EVEN = 0.524      # -110 both ways
MIN_EVIDENCE_PLAYS = 300


def research_only() -> bool:
    """True until the held-out record clears break-even on enough plays."""
    return not (EVIDENCE["plays"] >= MIN_EVIDENCE_PLAYS
                and EVIDENCE["hit_rate"] > BREAK_EVEN)


def gap_feature(projection: float, line: float) -> float:
    return (float(projection) - float(line)) / math.sqrt(max(float(line), 0.5))


def price(metric: str, projection: float | None, line: float | None) -> dict | None:
    """{'fair', 'p_over', 'family'} for one quote, or None if unpriceable."""
    family = FAMILY.get(metric)
    if family is None or projection is None or line is None:
        return None
    coef = COEFFICIENTS[family]
    z = coef["a"] + coef["b"] * gap_feature(projection, line)
    return {
        "family": family,
        "fair": round(float(line) + coef["w"] * (float(projection) - float(line)), 2),
        "p_over": round(1.0 / (1.0 + math.exp(-z)), 4),
    }
