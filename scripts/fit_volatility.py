"""Fit the team volatility ranking on the time-forward replay of the model.

Reuses `audit_regimes.expanding_predictions`: every game's margin and total come
from coefficients fit only on earlier seasons and from ratings and form built
only from earlier weeks, so a team's misses here are the misses the production
specification would have made at the time.

`volatility.fit` then measures, per market, whether a team's miss size against
the model is a trait (it carries from one half-season to the other, and into the
next season) or noise. The result -- with the last replayed season carried
as each team's prior -- goes to `reports/volatility_fit.json` and, as code an
installed package can import, `src/nflmodel/volatility_fit.py`.

Run: python scripts/fit_volatility.py   (needs numpy, like the other fit scripts)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import audit_regimes  # noqa: E402
import fit_matrix as fit  # noqa: E402

from nflmodel import ratings, volatility, volatility_data  # noqa: E402

REPORT = Path(__file__).resolve().parents[1] / "reports" / "volatility_fit.json"
MODULE = Path(__file__).resolve().parents[1] / "src" / "nflmodel" / "volatility_fit.py"


def graded_games() -> list[volatility.GradedGame]:
    schedule, lines = fit.load()
    rows = fit.build_features(schedule, lines)
    predictions = audit_regimes.expanding_predictions(rows)
    process = volatility_data.process_index(
        [line for season_lines in lines.values() for line in season_lines])
    # expanding_predictions walks the test seasons in order and each season's
    # rows in input order; rebuild that sequence to recover the team names.
    seasons = sorted({r["season"] for r in rows if r["season"] >= audit_regimes.FIRST_TEST_SEASON})
    tested = [r for s in seasons for r in rows if r["season"] == s]
    if len(tested) != len(predictions):
        raise SystemExit(f"replay mismatch: {len(tested)} rows, {len(predictions)} predictions")
    out = []
    for row, p in zip(tested, predictions):
        if (row["season"], row["week"]) != (p.season, p.week):
            raise SystemExit(f"replay misaligned at {row['season']} week {row['week']}")
        key = (p.season, p.week)
        out.append(volatility.GradedGame(
            season=p.season, week=p.week, home=row["home"], away=row["away"],
            model_margin=p.margin, actual_margin=p.actual_margin,
            model_total=p.total, actual_total=p.actual_total,
            home_process=process.get(key + (row["home"], row["away"])),
            away_process=process.get(key + (row["away"], row["home"])),
        ))
    return out


def main() -> int:
    games = graded_games()
    result = volatility.fit(
        games, margin_sd=ratings.MARGIN_SD,
        source=f"time-forward replay (audit_regimes), nflverse "
               f"{games[0].season}-{games[-1].season}",
        process_features=volatility_data.PROCESS_FEATURES,
        consistency_stats=volatility_data.CONSISTENCY_STATS,
    )
    REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    volatility.write_params_module(result, MODULE)
    print("\n".join(volatility.summary(result)))
    print(f"  wrote {REPORT} and {MODULE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
