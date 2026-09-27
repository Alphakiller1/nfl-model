"""Fit and audit the starting-quarterback availability adjustment, time-forward.

For every regular-season game 2020-2025 the point model's margin is rebuilt
exactly as `audit_regimes.expanding_predictions` builds it (coefficients from
earlier seasons only). Each team is then flagged if its usual starter was ruled
Out/Doubtful on that week's injury report or sat on a reserve/release roster
status -- the same information `availability.quarterback_status` uses live.

The coefficient scored in season S is fitted only on seasons before S, and the
report records:

* MAE of the model margin before and after, on all games and on flagged games;
* the market's own valuation (market margin minus model margin, regressed on the
  flag) as an independent cross-check -- the closing line knows who is starting;
* the final all-season coefficient to paste into `availability.QB_OUT_POINTS`.

Run: python scripts/fit_availability.py   (needs numpy; downloads ~30 nflverse files once)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import audit_regimes as audit
import fit_matrix as fit
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nflmodel import availability  # noqa: E402
from nflmodel.sources import nflverse  # noqa: E402

FIRST_SCORED = 2021  # 2020 has no earlier scored season to fit the coefficient on


def _season_rows(url: str, name: str) -> list[dict]:
    try:
        return nflverse.fetch_csv(url, name, ttl=None)
    except nflverse.NflverseError:
        return []


def _by_week(rows: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for row in rows:
        out.setdefault(int(nflverse.number(row.get("week")) or 0), []).append(row)
    return out


def predictions_with_teams(rows: list[dict]) -> list[dict]:
    """`audit.expanding_predictions`, keeping each game's identity."""
    out: list[dict] = []
    for test_season in sorted({int(r["season"]) for r in rows}):
        if test_season < audit.FIRST_TEST_SEASON:
            continue
        train = [r for r in rows if r["season"] < test_season]
        test = [r for r in rows if r["season"] == test_season]
        train_x, train_y, _ = fit.side_matrix(train)
        intercept, beta = fit._ols(train_x, train_y)
        for r in test:
            home = intercept + float(np.asarray(
                fit.side(r["home_form"], r["away_form"], not r["neutral"])) @ beta)
            away = intercept + float(np.asarray(
                fit.side(r["away_form"], r["home_form"], False)) @ beta)
            out.append({
                "season": test_season, "week": int(r["week"]),
                "home": r["home"], "away": r["away"],
                "actual": float(r["margin"]),
                "market": r["market_margin"],
                "model": 0.5 * float(r["base"]) + 0.5 * (home - away),
            })
    return out


def flag_games(games: list[dict]) -> None:
    seasons = sorted({g["season"] for g in games})
    for season in seasons:
        players = (_season_rows(nflverse.PLAYER_WEEK_URL.format(season=season - 1),
                                f"player_stats_week_{season - 1}.csv")
                   + _season_rows(nflverse.PLAYER_WEEK_URL.format(season=season),
                                  f"player_stats_week_{season}.csv"))
        injuries = _by_week(_season_rows(nflverse.INJURIES_URL.format(season=season),
                                         f"injuries_{season}.csv"))
        rosters = _by_week(_season_rows(nflverse.WEEKLY_ROSTER_URL.format(season=season),
                                        f"roster_weekly_{season}.csv"))
        if not injuries or not rosters:
            raise RuntimeError(f"{season}: injury report or weekly roster missing")
        for week in sorted({g["week"] for g in games if g["season"] == season}):
            starters = availability.usual_starters(players, season, week)
            status = availability.quarterback_status(
                starters, injuries.get(week, []), rosters.get(week, []))
            for g in games:
                if g["season"] == season and g["week"] == week:
                    g["home_out"] = availability.out_flag(status, g["home"])
                    g["away_out"] = availability.out_flag(status, g["away"])
                    g["flag"] = g["away_out"] - g["home_out"]
                    g["who"] = [s.starter_name for t, s in status.items()
                                if t in (g["home"], g["away"]) and not s.available]
        n = sum(1 for g in games if g["season"] == season and g["flag"] != 0)
        print(f"  {season}: {n} games with one side's starter unavailable")


def _slope(games: list[dict], target) -> tuple[float, float]:
    """No-intercept OLS of target on the flag, with its standard error."""
    x = np.array([g["flag"] for g in games], dtype=float)
    y = np.array([target(g) for g in games], dtype=float)
    keep = x != 0
    x, y = x[keep], y[keep]
    if len(x) < 5:
        return 0.0, float("inf")
    b = float(x @ y / (x @ x))
    resid = y - b * x
    se = float(np.sqrt(resid @ resid / (len(x) - 1) / (x @ x)))
    return b, se


def _mae(values) -> float:
    return statistics.fmean(abs(v) for v in values)


def run() -> dict:
    schedule, lines = fit.load()
    rows = fit.build_features(schedule, lines)
    games = predictions_with_teams(rows)
    flag_games(games)

    scored, folds = [], []
    for season in sorted({g["season"] for g in games}):
        if season < FIRST_SCORED:
            continue
        train = [g for g in games if g["season"] < season]
        b, se = _slope(train, lambda g: g["actual"] - g["model"])
        folds.append({"test_season": season, "points": round(b, 3), "se": round(se, 3)})
        for g in (x for x in games if x["season"] == season):
            scored.append({**g, "adjusted": g["model"] + b * g["flag"]})

    flagged = [g for g in scored if g["flag"] != 0]
    priced = [g for g in flagged if g["market"] is not None]
    outcome_b, outcome_se = _slope(games, lambda g: g["actual"] - g["model"])
    market_games = [g for g in games if g["market"] is not None]
    market_b, market_se = _slope(market_games, lambda g: g["market"] - g["model"])

    report = {
        "method": "time-forward; coefficient for season S fitted on seasons < S",
        "scored_seasons": [FIRST_SCORED, max(g["season"] for g in games)],
        "games": len(scored),
        "flagged_games": len(flagged),
        "folds": folds,
        "mae_all": {
            "model": round(_mae(g["actual"] - g["model"] for g in scored), 4),
            "model_plus_availability": round(_mae(g["actual"] - g["adjusted"] for g in scored), 4),
        },
        "mae_flagged": {
            "model": round(_mae(g["actual"] - g["model"] for g in flagged), 4),
            "model_plus_availability": round(
                _mae(g["actual"] - g["adjusted"] for g in flagged), 4),
            "market": round(_mae(g["actual"] - g["market"] for g in priced), 4) if priced else None,
        },
        "market_gap_flagged": {
            "model": round(_mae(g["model"] - g["market"] for g in priced), 4) if priced else None,
            "model_plus_availability": round(
                _mae(g["adjusted"] - g["market"] for g in priced), 4) if priced else None,
        },
        "outcome_points": {"all_seasons": round(outcome_b, 3), "se": round(outcome_se, 3)},
        "market_implied_points": {"all_seasons": round(market_b, 3), "se": round(market_se, 3)},
        "examples": [
            {k: g[k] for k in ("season", "week", "home", "away", "who")}
            for g in flagged[-8:]
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/availability_fit.json")
    args = parser.parse_args()
    report = run()
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))
    print(f"\nQB_OUT_POINTS = {report['outcome_points']['all_seasons']}"
          f"   (market values it at {report['market_implied_points']['all_seasons']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
