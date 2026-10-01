"""Outcome distributions for the player projection layer.

`player_props.project` publishes a mean for each stat. A reader comparing a
projection with a sportsbook line also needs its spread: how often a player
projected for 60 receiving yards finishes under 40 or over 90. This module
turns a mean into that distribution.

* Yardage and volume stats use an empirical ladder: the quantiles of
  actual / expected, measured on nflverse weekly player stats, bucketed by the
  size of the expectation (a small expectation has a much wider relative
  spread than a large one).
* Small counts (TDs, interceptions, receptions) use a negative binomial whose
  size was chosen for calibration at the lines books post (x.5).

Fitted by `scripts/fit_prop_distributions.py` (fit 2023-2024, scored on 2025,
shipped on all three). The expectation in that fit is each player's
recency-weighted in-season average, which is a weaker predictor than this
model's projection, so the bands are, if anything, slightly wide.
"""

from __future__ import annotations

import math

QUANTILE_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)

# metric -> [(bucket upper bound on the mean, ratio quantiles), ...]
RATIO_QUANTILES: dict[str, list[tuple[float, tuple[float, ...]]]] = {
    "pass_attempts": [
        (28, (0.202, 0.891, 1.199, 1.637, 3.118)),
        (34, (0.655, 0.813, 0.994, 1.179, 1.33)),
        (1e+09, (0.631, 0.774, 0.896, 1.066, 1.188)),
    ],
    "completions": [
        (18, (0.129, 0.837, 1.204, 1.75, 3.137)),
        (22, (0.6, 0.793, 1.006, 1.181, 1.337)),
        (1e+09, (0.606, 0.748, 0.92, 1.088, 1.259)),
    ],
    "passing_yards": [
        (200, (0.12, 0.744, 1.187, 1.668, 3.009)),
        (240, (0.573, 0.777, 1.025, 1.245, 1.447)),
        (1e+09, (0.561, 0.719, 0.926, 1.12, 1.298)),
    ],
    "rush_attempts": [
        (2.5, (0.0, 0.517, 1.077, 1.911, 3.03)),
        (4.5, (0.257, 0.56, 0.884, 1.331, 1.963)),
        (1e+09, (0.353, 0.584, 0.878, 1.204, 1.623)),
    ],
    "carries": [
        (6, (0.0, 0.198, 0.843, 1.892, 3.367)),
        (11, (0.279, 0.558, 0.933, 1.404, 1.924)),
        (16, (0.509, 0.715, 0.971, 1.248, 1.504)),
        (1e+09, (0.503, 0.708, 0.94, 1.165, 1.403)),
    ],
    "rushing_yards": [
        (10, (0.0, 0.0, 0.609, 2.673, 6.667)),
        (30, (0.0, 0.241, 0.781, 1.644, 2.685)),
        (55, (0.215, 0.49, 0.9, 1.42, 1.985)),
        (1e+09, (0.325, 0.551, 0.862, 1.22, 1.603)),
    ],
    "targets": [
        (3, (0.0, 0.0, 0.891, 1.742, 2.923)),
        (5, (0.28, 0.543, 0.912, 1.379, 1.829)),
        (7, (0.36, 0.615, 0.926, 1.321, 1.685)),
        (1e+09, (0.424, 0.655, 0.893, 1.157, 1.477)),
    ],
    "receiving_yards": [
        (20, (0.0, 0.0, 0.629, 1.992, 4.275)),
        (35, (0.0, 0.289, 0.781, 1.471, 2.273)),
        (55, (0.162, 0.43, 0.833, 1.377, 2.022)),
        (1e+09, (0.244, 0.495, 0.824, 1.228, 1.723)),
    ],
}

# metric -> negative binomial size (None = Poisson).
COUNT_SIZE: dict[str, float | None] = {"passing_tds": None, "interceptions": 2.0, "receptions": 8.0}


def _ladder(metric: str, mean: float) -> tuple[float, ...] | None:
    buckets = RATIO_QUANTILES.get(metric)
    if not buckets:
        return None
    for upper, quantiles in buckets:
        if mean <= upper:
            return quantiles
    return buckets[-1][1]


def _pmf(mean: float, size: float | None) -> dict[str, float]:
    out: dict[str, float] = {}
    cumulative = 0.0
    for k in range(0, 60):
        if mean <= 0:
            mass = 1.0 if k == 0 else 0.0
        elif size is None:
            mass = math.exp(-mean + k * math.log(mean) - math.lgamma(k + 1))
        else:
            p = size / (size + mean)
            mass = math.exp(math.lgamma(k + size) - math.lgamma(size) - math.lgamma(k + 1)
                            + size * math.log(p) + k * math.log(1 - p))
        cumulative += mass
        if mass >= 5e-4:
            out[str(k)] = round(mass, 4)
        if cumulative > 0.9995:
            break
    return out


def _pmf_quantile(pmf: dict[str, float], level: float) -> float:
    total = 0.0
    for k in sorted(pmf, key=int):
        total += pmf[k]
        if total >= level:
            return float(k)
    return float(max(pmf, key=int))


def distribution(metric: str, mean: float | None) -> dict | None:
    """Mean, 10th/50th/90th percentiles, and either a ``pmf`` (counts) or the
    quantile ladder ``q`` at ``q_levels`` (everything else). None when the
    metric has no fitted distribution."""
    if mean is None:
        return None
    mean = float(mean)
    if metric in COUNT_SIZE:
        pmf = _pmf(mean, COUNT_SIZE[metric])
        return {"mean": round(mean, 2), "p10": _pmf_quantile(pmf, 0.10),
                "p50": _pmf_quantile(pmf, 0.50), "p90": _pmf_quantile(pmf, 0.90), "pmf": pmf}
    ratios = _ladder(metric, mean)
    if ratios is None:
        return None
    ladder = [round(mean * r, 1) for r in ratios]
    return {"mean": round(mean, 2), "p10": ladder[0], "p50": ladder[2], "p90": ladder[-1],
            "q": ladder, "q_levels": list(QUANTILE_LEVELS)}


def over_probability(dist: dict, line: float) -> float | None:
    """P(stat > line): summed from the pmf, else read off the ladder."""
    if dist.get("pmf"):
        return sum(v for k, v in dist["pmf"].items() if float(k) > line)
    ladder, levels = dist.get("q"), dist.get("q_levels")
    if not ladder or not levels:
        return None
    if line < ladder[0]:
        return 1.0 - levels[0]
    if line >= ladder[-1]:
        return 1.0 - levels[-1]
    for (x0, p0), (x1, p1) in zip(zip(ladder, levels), zip(ladder[1:], levels[1:])):
        if x0 <= line < x1:
            return 1.0 - (p0 if x1 == x0 else p0 + (p1 - p0) * (line - x0) / (x1 - x0))
    return None


def stats(metrics: dict[str, float]) -> dict[str, dict]:
    """Every metric with a fitted distribution, keyed as `metrics` is."""
    out = {}
    for metric, mean in (metrics or {}).items():
        dist = distribution(metric, mean)
        if dist:
            out[metric] = dist
    return out
