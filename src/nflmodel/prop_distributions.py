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

Fitted on the projection layer's OWN errors: actual / projected for players
who played, from the 2025 point-in-time replay (research/props_model/
dist_fit.py). On the held-out second half of 2025 the 10-90 bands covered
81-84% for receiving and rushing (target 80%) and 70-77% for quarterbacks;
medians sat at 45-51% of outcomes. (The first version was fitted against each
player's trailing average, a weaker predictor than the projection, and was
far too wide for some quarterbacks: 20-512 passing yards around a 170 mean.)
"""

from __future__ import annotations

import math

QUANTILE_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)

# metric -> [(bucket upper bound on the mean, ratio quantiles), ...]
RATIO_QUANTILES: dict[str, list[tuple[float, tuple[float, ...]]]] = {
    "passing_yards": [
        (208.23, (0.481, 0.739, 1.019, 1.242, 1.404)),
        (220.42, (0.647, 0.776, 0.967, 1.209, 1.398)),
        (234.28, (0.621, 0.771, 0.997, 1.182, 1.405)),
        (1e09, (0.614, 0.773, 1.035, 1.181, 1.348)),
    ],
    "pass_attempts": [
        (29.7, (0.667, 0.798, 0.955, 1.16, 1.381)),
        (31.12, (0.657, 0.812, 0.984, 1.155, 1.28)),
        (32.47, (0.7, 0.855, 0.994, 1.192, 1.321)),
        (1e09, (0.707, 0.837, 1.003, 1.137, 1.355)),
    ],
    "completions": [
        (19.29, (0.578, 0.779, 0.968, 1.146, 1.297)),
        (20.22, (0.631, 0.796, 0.98, 1.178, 1.338)),
        (21.14, (0.686, 0.821, 0.978, 1.166, 1.378)),
        (1e09, (0.643, 0.758, 0.98, 1.174, 1.312)),
    ],
    "rushing_yards": [
        (12.52, (0.0, 0.0, 0.0, 0.991, 2.834)),
        (24.5, (0.0, 0.161, 0.628, 1.39, 2.275)),
        (48.77, (0.162, 0.422, 0.837, 1.357, 2.13)),
        (1e09, (0.369, 0.588, 0.923, 1.309, 1.713)),
    ],
    "carries": [
        (2.62, (0.0, 0.0, 0.4, 1.773, 3.77)),
        (7.35, (0.176, 0.339, 0.781, 1.32, 1.957)),
        (13.65, (0.415, 0.7, 0.939, 1.31, 1.607)),
        (1e09, (0.574, 0.768, 0.956, 1.21, 1.443)),
    ],
    "rush_attempts": [
        (3.05, (0.0, 0.354, 0.672, 1.069, 1.549)),
        (3.63, (0.276, 0.556, 0.867, 1.167, 1.63)),
        (4.69, (0.255, 0.508, 0.905, 1.546, 2.035)),
        (1e09, (0.412, 0.632, 0.951, 1.149, 1.523)),
    ],
    "targets": [
        (1.51, (0.0, 0.0, 0.833, 2.083, 3.368)),
        (2.9, (0.0, 0.51, 1.038, 1.613, 2.415)),
        (5.0, (0.299, 0.595, 0.962, 1.356, 1.805)),
        (1e09, (0.428, 0.663, 0.943, 1.286, 1.586)),
    ],
    "receiving_yards": [
        (10.22, (0.0, 0.0, 0.0, 1.612, 3.684)),
        (20.03, (0.0, 0.108, 0.8, 1.675, 2.879)),
        (37.45, (0.0, 0.323, 0.795, 1.458, 2.18)),
        (1e09, (0.263, 0.533, 0.89, 1.381, 1.872)),
    ],
}

# metric -> negative binomial size (None = Poisson).
COUNT_SIZE: dict[str, float | None] = {
    "passing_tds": None, "interceptions": None, "receptions": 12.0,
}


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
    """P(stat > line): summed from the pmf, else read off the ladder.

    Inside the ladder it interpolates between quantiles. Below the 10th
    percentile it runs linearly to 1.0 at zero; above the 90th it decays
    exponentially, at the rate set by the gap from the median to the 90th.
    """
    if dist.get("pmf"):
        return sum(v for k, v in dist["pmf"].items() if float(k) > line)
    ladder, levels = dist.get("q"), dist.get("q_levels")
    if not ladder or not levels:
        return None
    if line < ladder[0]:
        return 1.0 - levels[0] * line / ladder[0] if ladder[0] > 0 else 1.0 - levels[0]
    if line >= ladder[-1]:
        spread = max(ladder[-1] - ladder[2], 1.0)
        return (1.0 - levels[-1]) * math.exp(-(line - ladder[-1]) / spread)
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
