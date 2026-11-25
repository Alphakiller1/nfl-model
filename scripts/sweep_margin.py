"""Sweep the margin and total specification on future-season error AND on ATS.

`audit_regimes.py` scores one frozen specification. This sweeps the three
choices that specification makes, in the same strictly-expanding protocol --
coefficients fitted only on seasons preceding the season being scored:

* ``--halflife`` the recency decay in `efficiency.adjust`. It had none: the
  opponent adjustment was an unweighted mean over the whole sample while
  `ratings.build` decayed at an 8-week halflife, so the two halves of the
  published 50/50 blend disagreed about what "current" meant.
* ``--weights`` the share of the published margin taken from the efficiency
  path. `totals.EFFICIENCY_WEIGHT` is 0.50, selected on MAE.
* ``--shrink`` the total's shrink toward the league mean (`totals.TOTAL_SHRINK`).

**Why ATS is reported beside MAE.** The 50/50 blend was selected on MAE and it
is the variant with the *worse* ATS record: in `reports/regime_audit_2026.json`
the efficiency path alone goes 804-778 (50.82%) against the blend's 784-798
(49.56%) while being 0.10 points worse on MAE. The product is a side pick, so
MAE is the wrong objective to select on; a point forecast that is closer on
average can still be on the wrong side of the number more often.

Neither record is an edge: break-even at -110 is 52.38%. This script exists so
the choice is made against the objective that matters, and so a variant that
reaches break-even would be visible rather than averaged away.

    PYTHONPATH=src python scripts/sweep_margin.py --halflife none,4,6,8,12,16
    PYTHONPATH=src python scripts/sweep_margin.py --weights 0,0.25,0.5,0.75,1
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import fit_matrix as fit  # noqa: E402
import numpy as np  # noqa: E402

from nflmodel import efficiency  # noqa: E402

FIRST_TEST_SEASON = 2020
REPORT = ROOT / "reports" / "margin_sweep.json"


def _halflife(text: str) -> float | None:
    return None if text.strip().lower() in {"none", "off", ""} else float(text)


def predictions(rows: list[dict]) -> list[dict]:
    """Expanding-season points, carrying both margin paths unblended."""
    out: list[dict] = []
    for test_season in sorted({int(r["season"]) for r in rows}):
        if test_season < FIRST_TEST_SEASON:
            continue
        train = [r for r in rows if r["season"] < test_season]
        test = [r for r in rows if r["season"] == test_season]
        if not train or not test:
            continue
        league_total = statistics.fmean(float(r["total"]) for r in train)
        train_x, train_y, _ = fit.side_matrix(train)
        intercept, beta = fit._ols(train_x, train_y)
        for row in test:
            home = intercept + float(
                np.asarray(fit.side(row["home_form"], row["away_form"],
                                    not row["neutral"])) @ beta)
            away = intercept + float(
                np.asarray(fit.side(row["away_form"], row["home_form"], False)) @ beta)
            out.append({
                "season": test_season, "week": int(row["week"]),
                "actual_margin": float(row["margin"]), "actual_total": float(row["total"]),
                "market_margin": row["market_margin"], "market_total": row["market_total"],
                "rating_margin": float(row["base"]), "efficiency_margin": home - away,
                "raw_total": home + away, "league_total": league_total,
            })
    return out


def _ats(rows: list[dict], weight: float) -> dict:
    """Record against the closing spread for one blend weight."""
    wins = losses = pushes = 0
    dog = 0
    sided = 0
    errors = []
    for row in rows:
        model = weight * row["efficiency_margin"] + (1 - weight) * row["rating_margin"]
        errors.append(abs(model - row["actual_margin"]))
        market = row["market_margin"]
        if market is None:
            continue
        # nflverse `spread_line` is the home team's handicap, positive when the
        # home side is favoured; the model number is an expected home margin.
        market = float(market)
        if abs(model - market) < 1e-9:
            continue
        home_side = model > market
        sided += 1
        if (market > 0) != home_side:
            dog += 1
        cover = row["actual_margin"] - market
        if abs(cover) < 1e-9:
            pushes += 1
        elif (cover > 0) == home_side:
            wins += 1
        else:
            losses += 1
    played = wins + losses
    rate = wins / played if played else 0.0
    return {
        "mae": round(statistics.fmean(errors), 4) if errors else None,
        "ats": [wins, losses, pushes],
        "ats_rate": round(rate, 4),
        "ats_se": round(math.sqrt(rate * (1 - rate) / played), 4) if played else None,
        "underdog_side_share": round(dog / sided, 4) if sided else None,
    }


def _ou(rows: list[dict], shrink: float) -> dict:
    wins = losses = pushes = 0
    errors = []
    for row in rows:
        total = row["league_total"] + shrink * (row["raw_total"] - row["league_total"])
        errors.append(abs(total - row["actual_total"]))
        market = row["market_total"]
        if market is None or abs(total - float(market)) < 1e-9:
            continue
        over = total > float(market)
        edge = row["actual_total"] - float(market)
        if abs(edge) < 1e-9:
            pushes += 1
        elif (edge > 0) == over:
            wins += 1
        else:
            losses += 1
    played = wins + losses
    rate = wins / played if played else 0.0
    return {
        "mae": round(statistics.fmean(errors), 4) if errors else None,
        "ou": [wins, losses, pushes],
        "ou_rate": round(rate, 4),
        "ou_se": round(math.sqrt(rate * (1 - rate) / played), 4) if played else None,
    }


def market_baseline(rows: list[dict]) -> dict:
    margin = [abs(float(r["market_margin"]) - r["actual_margin"])
              for r in rows if r["market_margin"] is not None]
    total = [abs(float(r["market_total"]) - r["actual_total"])
             for r in rows if r["market_total"] is not None]
    return {"margin_mae": round(statistics.fmean(margin), 4) if margin else None,
            "total_mae": round(statistics.fmean(total), 4) if total else None,
            "games": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--halflife", default="none",
                        help="comma-separated efficiency halflives, or 'none'")
    parser.add_argument("--weights", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--shrink", default="0.5,0.6,0.7,0.8,0.9,1.0")
    args = parser.parse_args()

    schedule, lines = fit.load()
    report: dict = {"protocol": {
        "training": "strictly earlier seasons only",
        "first_test_season": FIRST_TEST_SEASON,
        "selection_note": ("MAE and ATS are reported together; the shipped 50/50 blend "
                           "was selected on MAE and has the worse ATS record"),
        "break_even_110": 0.5238,
    }, "halflives": {}}

    for text in args.halflife.split(","):
        halflife = _halflife(text)
        efficiency.HALFLIFE_WEEKS = halflife
        rows = fit.build_features(schedule, lines)
        preds = predictions(rows)
        label = "none" if halflife is None else f"{halflife:g}"
        entry = {"market": market_baseline(preds),
                 "margin": {}, "total": {}}
        for weight in (float(w) for w in args.weights.split(",")):
            entry["margin"][f"{weight:g}"] = _ats(preds, weight)
        for shrink in (float(s) for s in args.shrink.split(",")):
            entry["total"][f"{shrink:g}"] = _ou(preds, shrink)
        report["halflives"][label] = entry
        best_mae = min(entry["margin"].items(), key=lambda kv: kv[1]["mae"])
        best_ats = max(entry["margin"].items(), key=lambda kv: kv[1]["ats_rate"])
        print(f"halflife {label:>4s}: {len(preds)} games | "
              f"best MAE w={best_mae[0]} ({best_mae[1]['mae']}, ATS {best_mae[1]['ats_rate']}) | "
              f"best ATS w={best_ats[0]} ({best_ats[1]['ats_rate']}, MAE {best_ats[1]['mae']})",
              flush=True)

    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
