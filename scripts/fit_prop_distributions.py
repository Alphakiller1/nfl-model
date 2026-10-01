"""Fit the player-projection distributions (src/nflmodel/prop_distributions.py).

    python scripts/fit_prop_distributions.py            # report + constants
    python scripts/fit_prop_distributions.py --json reports/prop_distributions_fit.json

Data: nflverse weekly player stats, regular season 2023-2025. The expectation
for each player-game is his recency-weighted average (half-life four games)
over at least three earlier games that season - a stand-in for the model's
projection, which is not reproducible historically. Fit on 2023-2024, scored
on 2025 (10th-90th band coverage, over-probability calibration at x.5 lines),
then refit on all three for shipping. Needs pandas (dev only).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nflmodel import prop_distributions as pd_mod  # noqa: E402

URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{}.parquet"
SEASONS = (2023, 2024, 2025)
HALF_LIFE = 4.0
MIN_PRIOR = 3

# metric (as player_props publishes it) -> (nflverse column, positions)
METRICS = {
    "pass_attempts": ("attempts", ("QB",)),
    "completions": ("completions", ("QB",)),
    "passing_yards": ("passing_yards", ("QB",)),
    "passing_tds": ("passing_tds", ("QB",)),
    "interceptions": ("passing_interceptions", ("QB",)),
    "rush_attempts": ("carries", ("QB",)),
    "carries": ("carries", ("RB",)),
    "rushing_yards": ("rushing_yards", ("QB", "RB")),
    "receptions": ("receptions", ("RB", "WR", "TE")),
    "targets": ("targets", ("RB", "WR", "TE")),
    "receiving_yards": ("receiving_yards", ("RB", "WR", "TE")),
}
COUNTS = ("passing_tds", "interceptions", "receptions")
SIZES = (None, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
BUCKETS = {
    "pass_attempts": (28, 34), "completions": (18, 22), "passing_yards": (200, 240),
    "rush_attempts": (2.5, 4.5), "carries": (6, 11, 16), "rushing_yards": (10, 30, 55),
    "targets": (3, 5, 7), "receiving_yards": (20, 35, 55),
}


def rows() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        d = pd.read_parquet(URL.format(season))
        d = d[d["season_type"] == "REG"].copy()
        d["season"] = season
        frames.append(d)
    return pd.concat(frames)


def expectations(data: pd.DataFrame) -> pd.DataFrame:
    out = []
    data = data.sort_values(["season", "player_id", "week"])
    for metric, (column, positions) in METRICS.items():
        sub = data[data["position"].isin(positions)][["season", "player_id", "week", column]]
        for (_season, _pid), games in sub.groupby(["season", "player_id"]):
            values = games[column].fillna(0).to_numpy(float)
            weeks = games["week"].to_numpy()
            for i in range(MIN_PRIOR, len(values)):
                w = 0.5 ** (np.arange(i)[::-1] / HALF_LIFE)
                mean = float((values[:i] * w).sum() / w.sum())
                if mean <= 0:
                    continue
                out.append((metric, _season, weeks[i], mean, float(values[i])))
    return pd.DataFrame(out, columns=["metric", "season", "week", "mean", "actual"])


def fit_ratios(frame: pd.DataFrame) -> dict[str, list]:
    out = {}
    for metric, edges in BUCKETS.items():
        sub = frame[frame.metric == metric]
        buckets, lo = [], 0.0
        for hi in list(edges) + [1e9]:
            r = (sub[(sub["mean"] > lo) & (sub["mean"] <= hi)].eval("actual / mean")).to_numpy()
            if len(r) >= 60:
                buckets.append((hi, tuple(round(float(np.quantile(r, q)), 3)
                                          for q in pd_mod.QUANTILE_LEVELS)))
            lo = hi
        if buckets:
            out[metric] = buckets
    return out


def _lines(mean: float) -> list[float]:
    base = math.floor(mean)
    return [max(0.5, base - 0.5), base + 0.5, base + 1.5]


def fit_sizes(frame: pd.DataFrame) -> dict[str, float | None]:
    out = {}
    for metric in COUNTS:
        sub = frame[frame.metric == metric]
        best = None
        for size in SIZES:
            err = []
            for mean, actual in zip(sub["mean"], sub["actual"]):
                pmf = pd_mod._pmf(mean, size)
                for line in _lines(mean):
                    p = sum(v for k, v in pmf.items() if float(k) > line)
                    err.append((p - (actual > line)) ** 2)
            brier = float(np.mean(err))
            if best is None or brier < best[1]:
                best = (size, brier)
        out[metric] = best[0]
    return out


def score(frame: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for metric in METRICS:
        sub = frame[frame.metric == metric]
        inside = n = 0
        pred, obs = [], []
        for mean, actual in zip(sub["mean"], sub["actual"]):
            dist = pd_mod.distribution(metric, mean)
            if not dist:
                continue
            n += 1
            inside += dist["p10"] <= actual <= dist["p90"]
            line = math.floor(mean) + 0.5
            pred.append(pd_mod.over_probability(dist, line))
            obs.append(float(actual > line))
        if n:
            out[metric] = {"n": n, "band_10_90": round(inside / n, 3),
                           "over_pred": round(float(np.mean(pred)), 3),
                           "over_obs": round(float(np.mean(obs)), 3)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="reports/prop_distributions_fit.json")
    args = parser.parse_args()
    frame = expectations(rows())
    fit, test = frame[frame.season < 2025], frame[frame.season == 2025]
    pd_mod.RATIO_QUANTILES = fit_ratios(fit)
    pd_mod.COUNT_SIZE = fit_sizes(fit)
    report = score(test)
    print(f"fit 2023-2024 (n={len(fit)}) -> 2025 (n={len(test)})")
    for metric, r in report.items():
        print(f"  {metric:<16} n={r['n']:5d}  10-90 band {r['band_10_90']:.3f}  "
              f"P(over mean line) predicted {r['over_pred']:.3f} observed {r['over_obs']:.3f}")
    ship_ratios, ship_sizes = fit_ratios(frame), fit_sizes(frame)
    print("RATIO_QUANTILES =", json.dumps(ship_ratios))
    print("COUNT_SIZE =", json.dumps(ship_sizes))
    Path(args.json).write_text(json.dumps({
        "fit": [2023, 2024], "test": 2025, "test_report": report,
        "shipped": {"ratio_quantiles": ship_ratios, "count_size": ship_sizes},
    }, indent=1), encoding="utf-8")
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
