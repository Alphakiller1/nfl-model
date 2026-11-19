"""Select player corrections on 2025, freeze through 2025, test on 2026.

The 2024 first/second halves choose regularisation. 2025 accepts/rejects each
predeclared family. Final coefficients and outcome bands use only 2024-2025.
No 2026 result is used to choose coefficients, features or acceptance thresholds.
This is a retrospective time-forward audit, not a source-vintage betting replay.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nflmodel import prop_distributions, prop_intelligence  # noqa: E402

ROLE = ("role_trend", "role_cv", "zero_role_rate", "log_history", "new_team", "depth")
RATE = (
    "process_rate_gap",
    "completion_over_expectation",
    "recent_efficiency",
    "explosive_dependence",
    "man_response",
    "blitz_response",
    "play_action_response",
    "qb_yards_shift",
)
CATCH = (
    "completion_over_expectation",
    "expected_catch_gap",
    "recent_catch_gap",
    "new_team",
    "qb_completion_shift",
)
RUSH = ("rush_core_gap", "rush_explosive_dependence", "recent_rush_efficiency", "scramble_share")
TD = (
    "inside10_rate",
    "endzone_rate",
    "rush_inside5_rate",
    "rush_inside10_rate",
    "recent_efficiency",
    "log_history",
)
ALPHAS = (0.1, 1.0, 10.0)


def specifications() -> dict:
    out = {}
    for position in ("QB", "RB", "WR", "TE"):
        volume = "pass_attempts" if position == "QB" else "targets"
        out[f"{position}:{volume}"] = ("log", None, ROLE, 0.35)
        if position in {"QB", "RB"}:
            rush = "rush_attempts" if position == "QB" else "carries"
            out[f"{position}:{rush}"] = ("log", None, ROLE + ("scramble_share",), 0.40)
            out[f"{position}:rushing_yards"] = ("rate", rush, RUSH, 0.75)
        yardage = "passing_yards" if position == "QB" else "receiving_yards"
        catches = "completions" if position == "QB" else "receptions"
        out[f"{position}:{yardage}"] = ("rate", volume, RATE, 1.0)
        out[f"{position}:{catches}"] = ("rate", volume, CATCH, 0.10)
        if position == "QB":
            out[f"{position}:passing_tds"] = ("log", None, TD, 0.50)
            out[f"{position}:interceptions"] = ("log", None, RATE, 0.40)
        else:
            out[f"{position}:anytime_td_probability"] = ("cloglog", None, TD, 0.60)
    return out


def load(path: Path) -> list[dict]:
    return json.loads(gzip.decompress(path.read_bytes()))


def relevant(rows: list[dict], key: str) -> list[dict]:
    position, metric = key.split(":")
    return [
        r
        for r in rows
        if r["position"] == position and metric in r["metrics"] and metric in r["actual"]
    ]


def fit(rows: list[dict], key: str, spec: tuple, alpha: float) -> dict | None:
    link, exposure, features, limit = spec
    metric = key.split(":")[1]
    selected = relevant(rows, key)
    if exposure:
        selected = [
            r
            for r in selected
            if r["actual"].get(exposure, 0) > 0 and r["metrics"].get(exposure, 0) > 0
        ]
    if len(selected) < 80:
        return None
    raw = np.asarray([[r["features"].get(f, 0.0) for f in features] for r in selected], float)
    center, scale = raw.mean(axis=0), np.maximum(raw.std(axis=0), 0.05)
    x = np.column_stack([np.ones(len(raw)), (raw - center) / scale])
    prediction = np.array([r["metrics"][metric] for r in selected], float)
    y = np.array([r["actual"][metric] for r in selected], float)
    penalty = np.diag([0.02] + [alpha] * len(features))
    if link == "rate":
        opp = np.array([r["actual"][exposure] for r in selected], float)
        baseline_rate = prediction / np.array([r["metrics"][exposure] for r in selected])
        residual = y / opp - baseline_rate
        weight = opp / opp.sum()
        beta = np.linalg.solve(x.T @ (weight[:, None] * x) + penalty, x.T @ (weight * residual))
    else:
        offset = (
            np.maximum(prediction, 0.01)
            if link == "log"
            else -np.log1p(-np.clip(prediction, 0.0001, 0.9999))
        )

        def objective(beta):
            latent = np.clip(x @ beta, -3.0, 3.0)
            mu = offset * np.exp(latent)
            if link == "log":
                loss = mu - y * np.log(np.maximum(mu, 1e-9))
            else:
                p = np.clip(-np.expm1(-mu), 1e-7, 1 - 1e-7)
                loss = -y * np.log(p) - (1 - y) * np.log1p(-p)
            return float(loss.mean() + 0.5 * beta @ penalty @ beta)

        optimized = minimize(objective, np.zeros(x.shape[1]), method="BFGS")
        beta = optimized.x
    return {
        "link": link,
        "exposure": exposure,
        "features": list(features),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "intercept": float(beta[0]),
        "coefficients": beta[1:].tolist(),
        "limit": limit,
        "alpha": alpha,
        "fit_rows": len(selected),
        "enabled": True,
    }


def score(rows: list[dict], key: str) -> dict:
    values = relevant(rows, key)
    metric = key.split(":")[1]
    if not values:
        return {"n": 0, "mse": None}
    errors = np.array([r["metrics"][metric] - r["actual"][metric] for r in values])
    return {
        "n": len(values),
        "mse": float(np.mean(errors**2)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(abs(errors))),
        "bias": float(errors.mean()),
    }


def corrected(rows, specs):
    return prop_intelligence.correct_metrics(rows, {"corrections": specs})


def uncertainty(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        # Prop outcomes are conditional on participating; a true DNP is voided.
        if not row["played"]:
            continue
        for metric, mean in row["metrics"].items():
            if mean > 0.1 and metric in row["actual"] and metric != "anytime_td_probability":
                groups[f"{row['position']}:{metric}"].append((row, mean, row["actual"][metric]))
    out = {}
    for key, group in groups.items():
        if len(group) < 100:
            continue
        edges = np.quantile([m for _, m, _ in group], [0.25, 0.5, 0.75]).tolist() + [1e9]
        cohorts = {}
        for cohort in ("all", "stable", "uncertain"):
            selected = [
                (r, m, a)
                for r, m, a in group
                if cohort == "all"
                or (
                    (r["features"]["role_cv"] > 0.5 or r["evidence"]["current_stint_games"] < 4)
                    == (cohort == "uncertain")
                )
            ]
            if len(selected) < 120:
                continue
            buckets, lower = [], -1.0
            for upper in edges:
                part = [(m, a) for _, m, a in selected if lower < m <= upper]
                if len(part) < 35:
                    part = [(m, a) for _, m, a in selected]
                residuals = [(a - m) / math.sqrt(max(m, 1)) for m, a in part]
                quantiles = np.quantile(residuals, prop_distributions.QUANTILE_LEVELS).tolist()
                buckets.append({"upper": upper, "n": len(part), "residual_quantiles": quantiles})
                lower = upper
            cohorts[cohort] = buckets
        out[key] = {
            "n": len(group),
            "cohorts": cohorts,
            "basis": "pre-2026 projection errors conditional on a recorded player appearance",
        }
        metric = key.split(":")[1]
        if metric in {
            "pass_attempts",
            "completions",
            "passing_tds",
            "interceptions",
            "targets",
            "receptions",
            "carries",
            "rush_attempts",
            "fg_attempts",
            "fg_made",
            "pat_made",
        }:
            count_sizes = {}
            for cohort in cohorts:
                selected = [
                    (r, m, a)
                    for r, m, a in group
                    if cohort == "all"
                    or (
                        (r["features"]["role_cv"] > 0.5 or r["evidence"]["current_stint_games"] < 4)
                        == (cohort == "uncertain")
                    )
                ]
                means = np.array([m for _, m, _ in selected])
                actuals = np.array([a for _, _, a in selected])
                choices = []
                for size in (None, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0):
                    logmass = (
                        poisson.logpmf(actuals, means)
                        if size is None
                        else nbinom.logpmf(actuals, size, size / (size + means))
                    )
                    choices.append((-float(logmass.mean()), size))
                count_sizes[cohort] = min(choices, key=lambda c: c[0])[1]
            out[key]["count_sizes"] = count_sizes
    return out


def score_bands(rows: list[dict], old_rows: list[dict]) -> dict:
    original = {(r["player_id"], r["game_id"]): r for r in old_rows}
    values = defaultdict(lambda: {"legacy": [], "revised": []})
    for row in rows:
        if not row["played"]:
            continue
        old = original.get((row["player_id"], row["game_id"]))
        if old is None:
            continue
        for metric, actual in row["actual"].items():
            if metric not in row["metrics"] or metric not in old["metrics"]:
                continue
            key = f"{row['position']}:{metric}"
            for label, mean, kwargs in (
                ("legacy", old["metrics"][metric], {}),
                (
                    "revised",
                    row["metrics"][metric],
                    {"position": row["position"], "evidence": row["evidence"]},
                ),
            ):
                dist = prop_distributions.distribution(metric, mean, **kwargs)
                if not dist:
                    continue
                lo, hi = dist["p10"], dist["p90"]
                width = hi - lo
                interval_score = width + 10 * max(lo - actual, 0) + 10 * max(actual - hi, 0)
                values[key][label].append(
                    (lo <= actual <= hi, width, interval_score, abs(dist["p50"] - actual))
                )
    out = {}
    for key, sides in values.items():
        out[key] = {}
        for label, records in sides.items():
            if not records:
                continue
            array = np.array(records, dtype=float)
            out[key][label] = {
                "n": len(records),
                "coverage_80": float(array[:, 0].mean()),
                "mean_width": float(array[:, 1].mean()),
                "interval_score_80": float(array[:, 2].mean()),
                "median_mae": float(array[:, 3].mean()),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--parameters", type=Path, default=ROOT / "src/nflmodel/prop_parameters.json"
    )
    args = parser.parse_args()
    data = {year: load(args.rows / f"structural-{year}.json.gz") for year in (2024, 2025, 2026)}
    import csv

    for year, rows in data.items():
        history = []
        for season in range(year - 3, year + 1):
            with (args.cache / f"player_stats_week_{season}.csv").open(encoding="utf-8-sig") as f:
                history.extend(csv.DictReader(f))
        for week in {r["week"] for r in rows}:
            totals = prop_intelligence.team_totals(history, season=year, week=week)
            prop_intelligence.add_quarterback_features(
                [r for r in rows if r["week"] == week], totals, season=year, week=week
            )
    tune_train = [r for r in data[2024] if r["week"] <= 10]
    tune_test = [r for r in data[2024] if r["week"] > 10]
    accepted, decisions = {}, {}
    for key, spec in specifications().items():
        choices = []
        for alpha in ALPHAS:
            fitted = fit(tune_train, key, spec, alpha)
            if fitted:
                result = score(corrected(tune_test, {key: fitted}), key)
                choices.append((result["mse"], alpha))
        if not choices:
            continue
        alpha = min(choices)[1]
        candidate = fit(data[2024], key, spec, alpha)
        if not candidate:
            continue
        reference = corrected(data[2025], accepted)
        challenger = corrected(data[2025], {**accepted, key: candidate})
        before, after = score(reference, key), score(challenger, key)
        gain = 1 - after["mse"] / before["mse"] if before["mse"] else 0.0
        halves = []
        for first in (True, False):
            old = score([r for r in reference if (r["week"] <= 10) == first], key)
            new = score([r for r in challenger if (r["week"] <= 10) == first], key)
            halves.append(1 - new["mse"] / old["mse"] if old["mse"] else 0.0)
        keep = before["n"] >= 100 and gain >= 0.0025 and min(halves) >= -0.01
        decisions[key] = {
            "enabled": keep,
            "validation_2025": {
                "baseline": before,
                "candidate": after,
                "mse_improvement": gain,
                "half_season_improvements": halves,
            },
            "alpha": alpha,
        }
        if keep:
            accepted[key] = candidate
        print(key, "ACCEPT" if keep else "REJECT", f"2025 MSE change {gain:+.2%}", flush=True)
    final = {
        key: fit(data[2024] + data[2025], key, specifications()[key], fitted["alpha"])
        for key, fitted in accepted.items()
    }
    params = {
        "version": prop_intelligence.VERSION,
        "model_version": "nfl-player-projections/3.0.0",
        "fit_seasons": [2024, 2025],
        "selection_season": 2025,
        "held_out_season": 2026,
        "fit_season_max": 2025,
        "corrections": final,
        "state": "time_forward_research",
        "validation_summary": {key: decisions[key] for key in final},
        "acceptance": "2025 MSE improves >=0.25%; neither half worsens >1%; n>=100",
        "uncertainty": uncertainty(corrected(data[2024] + data[2025], final)),
    }
    args.parameters.write_text(json.dumps(params, indent=2, allow_nan=False), encoding="utf-8")
    prop_intelligence.parameters.cache_clear()
    report = {"protocol": __doc__, "decisions": decisions, "seasons": {}}
    for year, rows in data.items():
        updated = corrected(rows, final)
        original = load(args.baseline / f"structural-{year}.json.gz")
        (args.out.parent / f"revised-{year}.json.gz").write_bytes(
            gzip.compress(
                json.dumps(updated, separators=(",", ":"), allow_nan=False).encode(), mtime=0
            )
        )
        report["seasons"][str(year)] = {
            key: {
                "published_v2_structure": score(original, key),
                "revised_structure": score(rows, key),
                "revised": score(updated, key),
            }
            for key in specifications()
            if relevant(rows, key)
        }
        if year == 2026:
            report["held_out_uncertainty_2026"] = score_bands(updated, original)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
