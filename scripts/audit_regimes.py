"""Time-forward audit of NFL margin, total, and favorite/underdog behavior.

The main fit report uses leave-one-season-out coefficients.  That is useful for
isolating every scored season from its own outcomes, but it is not a historical
simulation because an early held-out season can still borrow coefficients from
later seasons.  This audit is stricter: every prediction is fit only on earlier
seasons, and every calibration candidate is learned only from earlier
predictions.  It freezes the 2026 feature specification and hyperparameters,
which were selected in the legacy leave-one-season-out study; it is therefore a
time-forward coefficient audit, not a fully nested or prospective trial.

No candidate is promoted merely because it changes the visual balance of the
board.  The report records MAE, market-relative decisions, favorite/underdog
direction, and fold stability so a regime transform must earn its place on
future-outcome error.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import fit_matrix as fit
import numpy as np

FIRST_TEST_SEASON = 2020
BUCKETS = {
    "weeks_1_4": (1, 4),
    "weeks_5_9": (5, 9),
    "weeks_10_18": (10, 18),
}


@dataclass(frozen=True)
class Prediction:
    season: int
    week: int
    actual_margin: float
    actual_total: float
    market_margin: float | None
    market_total: float | None
    rating_margin: float
    efficiency_margin: float
    margin: float
    league_total: float
    total: float


def _bucket(week: int) -> str:
    for name, (low, high) in BUCKETS.items():
        if low <= week <= high:
            return name
    raise ValueError(f"unsupported regular-season week: {week}")


def _ols(features: list[list[float]], targets: list[float]) -> np.ndarray:
    matrix = np.column_stack([np.ones(len(features)), np.asarray(features, dtype=float)])
    beta, *_ = np.linalg.lstsq(matrix, np.asarray(targets, dtype=float), rcond=None)
    return beta


def _apply(beta: np.ndarray, *features: float) -> float:
    return float(beta[0] + np.asarray(features, dtype=float) @ beta[1:])


def expanding_predictions(rows: list[dict]) -> list[Prediction]:
    """Fit the point model only on seasons preceding the season being scored."""
    out: list[Prediction] = []
    seasons = sorted({int(row["season"]) for row in rows})
    for test_season in seasons:
        if test_season < FIRST_TEST_SEASON:
            continue
        train = [row for row in rows if row["season"] < test_season]
        test = [row for row in rows if row["season"] == test_season]
        if not train or not test:
            continue
        league_total = statistics.fmean(float(row["total"]) for row in train)
        train_x, train_y, _ = fit.side_matrix(train)
        intercept, beta = fit._ols(train_x, train_y)
        for row in test:
            home_points = intercept + float(
                np.asarray(
                    fit.side(row["home_form"], row["away_form"], not row["neutral"])
                )
                @ beta
            )
            away_points = intercept + float(
                np.asarray(fit.side(row["away_form"], row["home_form"], False)) @ beta
            )
            efficiency_margin = home_points - away_points
            margin = 0.5 * float(row["base"]) + 0.5 * efficiency_margin
            raw_total = home_points + away_points
            total = league_total + 0.70 * (raw_total - league_total)
            out.append(
                Prediction(
                    season=test_season,
                    week=int(row["week"]),
                    actual_margin=float(row["margin"]),
                    actual_total=float(row["total"]),
                    market_margin=row["market_margin"],
                    market_total=row["market_total"],
                    rating_margin=float(row["base"]),
                    efficiency_margin=efficiency_margin,
                    margin=margin,
                    league_total=league_total,
                    total=total,
                )
            )
    return out


def _time_forward_transform(
    rows: list[Prediction],
    *,
    target: str,
    features: tuple[str, ...],
    by_week: bool = False,
    by_bucket: bool = False,
) -> tuple[list[float], list[dict]]:
    """Fit a candidate transform on earlier prediction seasons only."""
    predictions: list[float] = []
    folds: list[dict] = []
    for row in rows:
        train = [candidate for candidate in rows if candidate.season < row.season]
        if by_week:
            train = [candidate for candidate in train if candidate.week == row.week]
        elif by_bucket:
            train = [
                candidate for candidate in train
                if _bucket(candidate.week) == _bucket(row.week)
            ]
        # The first scored expanding season has no earlier scored fold from
        # which to learn a second-stage transform. Preserve the baseline rather
        # than fitting on the outcome being predicted.
        if len(train) < max(32, len(features) * 12):
            predictions.append(float(getattr(row, target.replace("actual_", ""))))
            continue
        x = [[float(getattr(candidate, name)) for name in features] for candidate in train]
        y = [float(getattr(candidate, target)) for candidate in train]
        beta = _ols(x, y)
        predictions.append(_apply(beta, *(float(getattr(row, name)) for name in features)))
        folds.append(
            {
                "test_season": row.season,
                "regime": str(row.week) if by_week else _bucket(row.week) if by_bucket else "all",
                "n_train": len(train),
                "intercept": float(beta[0]),
                "coefficients": {
                    name: float(value) for name, value in zip(features, beta[1:], strict=True)
                },
            }
        )
    return predictions, _dedupe_folds(folds)


def _dedupe_folds(folds: list[dict]) -> list[dict]:
    unique: dict[tuple, dict] = {}
    for fold in folds:
        key = (fold["test_season"], fold["regime"], *fold["coefficients"])
        unique[key] = fold
    return list(unique.values())


def _score_margin(rows: list[Prediction], predicted: list[float]) -> dict:
    errors = [abs(row.actual_margin - value) for row, value in zip(rows, predicted, strict=True)]
    residuals = [
        row.actual_margin - value for row, value in zip(rows, predicted, strict=True)
    ]
    calibration = _ols([[value] for value in predicted], [row.actual_margin for row in rows])
    wins = losses = pushes = dogs = favorites = 0
    for row, value in zip(rows, predicted, strict=True):
        market = row.market_margin
        if market is None or abs(value - market) < 1e-9:
            continue
        model_side = 1 if value > market else -1
        actual_edge = row.actual_margin - market
        if abs(actual_edge) < 1e-9:
            pushes += 1
        elif actual_edge * model_side > 0:
            wins += 1
        else:
            losses += 1
        if abs(market) > 1e-9 and model_side * market < 0:
            dogs += 1
        else:
            favorites += 1
    decisions = wins + losses
    rate = wins / decisions if decisions else None
    return {
        "n": len(rows),
        "mae": statistics.fmean(errors),
        "ats": [wins, losses, pushes],
        "ats_rate": rate,
        "underdog_side_share": dogs / (dogs + favorites) if dogs + favorites else None,
        "residual_sd": statistics.stdev(residuals),
        "calibration_intercept": float(calibration[0]),
        "calibration_slope": float(calibration[1]),
    }


def _score_total(rows: list[Prediction], predicted: list[float]) -> dict:
    errors = [abs(row.actual_total - value) for row, value in zip(rows, predicted, strict=True)]
    residuals = [row.actual_total - value for row, value in zip(rows, predicted, strict=True)]
    calibration = _ols([[value] for value in predicted], [row.actual_total for row in rows])
    wins = losses = pushes = 0
    for row, value in zip(rows, predicted, strict=True):
        market = row.market_total
        if market is None or abs(value - market) < 1e-9:
            continue
        actual_edge = row.actual_total - market
        direction = 1 if value > market else -1
        if abs(actual_edge) < 1e-9:
            pushes += 1
        elif actual_edge * direction > 0:
            wins += 1
        else:
            losses += 1
    decisions = wins + losses
    return {
        "n": len(rows),
        "mae": statistics.fmean(errors),
        "ou": [wins, losses, pushes],
        "ou_rate": wins / decisions if decisions else None,
        "residual_sd": statistics.stdev(residuals),
        "calibration_intercept": float(calibration[0]),
        "calibration_slope": float(calibration[1]),
    }


def audit() -> dict:
    schedule, lines = fit.load()
    rows = fit.build_features(schedule, lines)
    expanding = expanding_predictions(rows)
    if not expanding:
        raise RuntimeError("no expanding-season predictions were produced")

    margin_candidates: dict[str, tuple[list[float], list[dict]]] = {
        "rating_only": ([row.rating_margin for row in expanding], []),
        "efficiency_only": ([row.efficiency_margin for row in expanding], []),
        "baseline_50_50": ([row.margin for row in expanding], []),
        "global_affine": _time_forward_transform(
            expanding, target="actual_margin", features=("margin",)
        ),
        "bucket_affine": _time_forward_transform(
            expanding, target="actual_margin", features=("margin",), by_bucket=True
        ),
        "week_affine": _time_forward_transform(
            expanding, target="actual_margin", features=("margin",), by_week=True
        ),
        "bucket_component_fit": _time_forward_transform(
            expanding,
            target="actual_margin",
            features=("rating_margin", "efficiency_margin"),
            by_bucket=True,
        ),
    }
    total_candidates: dict[str, tuple[list[float], list[dict]]] = {
        "training_league_mean": ([row.league_total for row in expanding], []),
        "baseline_shrunk": ([row.total for row in expanding], []),
        "global_affine": _time_forward_transform(
            expanding, target="actual_total", features=("total",)
        ),
        "bucket_affine": _time_forward_transform(
            expanding, target="actual_total", features=("total",), by_bucket=True
        ),
        "week_affine": _time_forward_transform(
            expanding, target="actual_total", features=("total",), by_week=True
        ),
    }

    market_margin_rows = [row for row in expanding if row.market_margin is not None]
    market_total_rows = [row for row in expanding if row.market_total is not None]
    payload = {
        "protocol": {
            "training": "strictly earlier seasons only",
            "specification": (
                "frozen 2026 feature specification and hyperparameters; "
                "coefficients and candidate transforms are time-forward"
            ),
            "hyperparameter_caveat": (
                "legacy leave-one-season-out selection is not nested inside each fold"
            ),
            "prospective_status": "retrospective audit; not a locked prospective trial",
            "first_test_season": FIRST_TEST_SEASON,
            "last_test_season": max(row.season for row in expanding),
            "games": len(expanding),
        },
        "margin": {
            name: {**_score_margin(expanding, values), "folds": folds}
            for name, (values, folds) in margin_candidates.items()
        },
        "total": {
            name: {**_score_total(expanding, values), "folds": folds}
            for name, (values, folds) in total_candidates.items()
        },
        "market": {
            "margin_mae": statistics.fmean(
                abs(row.actual_margin - float(row.market_margin)) for row in market_margin_rows
            ),
            "total_mae": statistics.fmean(
                abs(row.actual_total - float(row.market_total)) for row in market_total_rows
            ),
        },
        "by_regime": {},
    }
    for name, (low, high) in BUCKETS.items():
        selected = [row for row in expanding if low <= row.week <= high]
        payload["by_regime"][name] = {
            "baseline_margin": _score_margin(selected, [row.margin for row in selected]),
            "baseline_total": _score_total(selected, [row.total for row in selected]),
            "market_margin_mae": statistics.fmean(
                abs(row.actual_margin - float(row.market_margin))
                for row in selected
                if row.market_margin is not None
            ),
            "market_total_mae": statistics.fmean(
                abs(row.actual_total - float(row.market_total))
                for row in selected
                if row.market_total is not None
            ),
        }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/regime_audit_2026.json")
    args = parser.parse_args()
    payload = audit()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "protocol": payload["protocol"],
        "margin": {name: {key: value for key, value in result.items() if key != "folds"}
                   for name, result in payload["margin"].items()},
        "total": {name: {key: value for key, value in result.items() if key != "folds"}
                  for name, result in payload["total"].items()},
        "market": payload["market"],
    }, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
