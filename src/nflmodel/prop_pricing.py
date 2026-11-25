"""Price a player prop from the projection matrix, calibrated against the line.

Each quote is priced from the projection itself, not from a market base
rate:

1. ``raw`` = P(stat > line) under the outcome distribution fitted on the
   projection layer's own errors (`prop_distributions`), centred on the
   matrix projection.
2. ``p_over`` = 0.5 + s * (raw - c). The raw distribution is honest about the
   player's spread of outcomes, but it ignores what the line knows. Against
   DraftKings closing lines a raw 70% hit well under that, so it is shrunk
   toward even by a factor s fitted leave-one-week-out. c is the raw value
   that corresponds to an even line.
3. ``fair`` = line + w * (projection - line): how far toward the projection
   the number should move, by market family.

Fitted on the current forecaster, not its predecessor
-----------------------------------------------------
`CALIBRATED_PROJECTION_VERSION` exists because a betting record earned by one
forecaster does not qualify another. It was doing its job and nobody noticed:
the projection layer moved to ``3.0.0`` while this file still named ``2.1.0``,
so `calibration_applies` returned False, `price` returned None for **every**
prop, and the board published no prop comparison, no prop best bet and no slip.
A version string was silently setting policy.

The fix was to earn the calibration rather than edit the string.
`scripts/fit_prop_pricing.py --replay` now joins a point-in-time v3 replay to
the closing DraftKings lines ESPN publishes for finished games, so a new version
can be calibrated without waiting a season for the ledger to fill. Refit on
2026 weeks 2-11 -- 6,346 graded lines against the previous fit's 1,920:

                        old (v2.1.0, wk 1-3)      new (v3.1.0, wk 2-11)
    held-out plays      838                       2,729
    hit rate            53.70%                    55.11%
    90% lower bound     51.49%                    53.89%
    held-out Brier      0.2499                    0.2487
    ranked record       54.78% (1,256)            52.38% (4,160)

Break-even at -110 is 52.38%. The held-out record clears it with a one-sided
90% lower bound of 53.86%, on three times the sample, so `research_only` is
satisfied on evidence rather than by assertion.

**The passing blend weight fitted to zero.** `BLEND_WEIGHT["passing"] = 0.0`
means the fair number for a passing market is the line: against the closing
line, this model's quarterback projections carry no information about where the
number should sit. It is left at the fitted value rather than floored at
something that looks more useful, and it is the clearest signal in the fit of
where the projection layer needs work.

Which book a calibration applies to
-----------------------------------
`raw` depends on the projection; the *map* from raw to a probability was fitted
where the line is a DraftKings closing number. `sources/bovada.py` now supplies
most pregame quotes, and Bovada is a softer book posting a line earlier. The
shrink calibrates the model's own overconfidence against the sharpest line
available, so applying it to a softer line errs toward even -- conservative, in
a known direction, rather than optimistic. That is defensible enough to publish
and not good enough to leave unmarked, so a quote from a book outside
`CALIBRATED_BOOKS` is priced with ``calibration_transfer`` naming the book, and
the forward record can settle whether the transfer holds.
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
# scripts/fit_prop_pricing.py --replay <dir> --season 2026 (weeks 2-11).
BLEND_WEIGHT = {"passing": 0.0, "receiving": 0.406, "rushing": 0.191}
CALIBRATION = {"s": 0.25, "c": 0.56}
EVIDENCE = {
    "source": "2026 weeks 2-11, point-in-time v3.1.0 replay vs DraftKings closing lines",
    "lines": 6346, "plays": 2729, "wins": 1504, "hit_rate": 0.551,
    "ranked_wins": 2179, "ranked_plays": 4160,
}
MIN_PROBABILITY = 0.53
BREAK_EVEN = 0.524      # -110 both ways
MIN_EVIDENCE_PLAYS = 300
CALIBRATED_PROJECTION_VERSION = "nfl-player-projections/3.1.0"
# Books whose lines the calibration above was actually fitted against.
CALIBRATED_BOOKS = frozenset({"draftkings"})


def calibration_applies(model_version: str | None) -> bool:
    return not model_version or model_version == CALIBRATED_PROJECTION_VERSION


def book_transfer(book: str | None) -> str | None:
    """The book this quote came from, when it is not one the fit used."""
    key = str(book or "").strip().lower()
    if not key or key in CALIBRATED_BOOKS:
        return None
    return key


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


def price(metric: str, projection: float | None, line: float | None, *, player=None,
          book: str | None = None) -> dict | None:
    """{'family', 'fair', 'raw', 'p_over'} for one quote, or None if unpriceable.

    ``calibration_transfer`` is present and names the book when the quote came
    from one the calibration was not fitted against.
    """
    family = FAMILY.get(metric)
    if player is not None and not calibration_applies(getattr(player, "model_version", None)):
        # One forecaster's betting record cannot qualify a different one. The
        # projection, its outcome bands and the raw threshold probability stay
        # available; a calibrated price needs this version's own record.
        return None
    if family is None or projection is None or line is None or projection <= 0:
        return None
    dist = prop_distributions.distribution(metric, projection)
    raw = prop_distributions.over_probability(dist, float(line)) if dist else None
    if raw is None:
        return None
    priced = {
        "family": family,
        "fair": round(float(line) + BLEND_WEIGHT[family] * (float(projection) - float(line)), 2),
        "raw": round(raw, 4),
        "p_over": round(calibrate(raw), 4),
    }
    transfer = book_transfer(book)
    if transfer is not None:
        priced["calibration_transfer"] = transfer
    return priced
