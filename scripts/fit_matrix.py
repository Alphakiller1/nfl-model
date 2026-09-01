"""Fit the logic matrix, the totals model and the rating constants.

Everything in `matrix.py`, `totals.py`, and the constants at the top of
`ratings.py` and `preseason.py` is produced here. It is a development script, not
part of the shipped package: it may use numpy, while the package itself stays
dependency-free and carries only the fitted numbers.

Specification
-------------
The model predicts **points scored by one team in one game**, from that team's
opponent-adjusted offence and the opponent's opponent-adjusted defence:

    points(A vs B) = c0 + SUM_x  g_x * ( off_x(A) + def_x(B) )  + h * is_home

`off_x` and `def_x` measure the same per-play quantity in the same units -- what
A's offence produced, and what B's defence allowed -- so a matchup is their
*sum* and one coefficient per family suffices. An earlier version gave the two
sides free coefficients and produced sign-flipped nonsense: `def_epa` fitted
POSITIVE (allowing more EPA per play improved the margin) because `epa` and
`first_down` correlate at r = 0.83 and the pair split the effect with opposite
signs. The symmetric form costs 0.016 points of MAE -- noise -- and every
coefficient it produces has the sign football says it should.

Margin and total both fall out of that single fit, and so do the offensive and
defensive power rankings:

    offence_index(A) =  SUM_x g_x * ( off_x(A) - mean_x )   points generated
    defence_index(B) = -SUM_x g_x * ( def_x(B) - mean_x )   points prevented
    efficiency margin = (off+def)(home) - (off+def)(away) + h

Everything is strictly point-in-time: for a game in season S week W, features are
built from completed games before W in S, blended by `preseason.py` with earlier
seasons. Coefficients are fitted **leave-one-season-out**, so no game is scored
by a model that saw it -- the only fit whose MAE is comparable with the market's.

    python scripts/fit_matrix.py                # full fit + sweeps
    python scripts/fit_matrix.py --no-sweeps    # coefficients only
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nflmodel import efficiency, preseason, ratings, teams, totals  # noqa: E402
from nflmodel.sources import nflverse  # noqa: E402

FIRST_TARGET = 2017
LAST_TARGET = 2025
FIRST_LOADED = FIRST_TARGET - 3
STATS = efficiency.ADJUSTABLE_STATS


# -- data ---------------------------------------------------------------------
def load() -> tuple[list[dict], dict[int, list[efficiency.GameLine]]]:
    seasons = tuple(range(FIRST_LOADED, LAST_TARGET + 1))
    schedule = nflverse.games(seasons=seasons)
    lines: dict[int, list[efficiency.GameLine]] = {}
    for season in seasons:
        rows = nflverse.team_week(season, completed_season=True)
        lines[season] = efficiency.game_lines(rows)
    print(f"  {len(schedule)} scheduled games, "
          f"{sum(len(v) for v in lines.values())} team-game lines "
          f"{FIRST_LOADED}-{LAST_TARGET}")
    return schedule, lines


def build_features(schedule: list[dict],
                   lines: dict[int, list[efficiency.GameLine]]) -> list[dict]:
    """One row per completed regular-season game in the target window."""
    history = ratings.from_rows(schedule)
    all_lines = [line for season_lines in lines.values() for line in season_lines]
    rows: list[dict] = []

    for season in range(FIRST_TARGET, LAST_TARGET + 1):
        prior_ratings = preseason.rating_prior(history, season)
        prior_forms = preseason.form_prior(all_lines, season)
        season_games = [g for g in history if g.season == season]
        season_lines = lines.get(season, [])

        for week in sorted({g.week for g in season_games}):
            before = [g for g in season_games if g.week < week]
            before_lines = [line for line in season_lines if line.week < week]
            played: dict[str, float] = {}
            for g in before:
                played[g.home] = played.get(g.home, 0.0) + 1
                played[g.away] = played.get(g.away, 0.0) + 1
            table = preseason.blend_ratings(prior_ratings, ratings.build(before), played)
            forms = preseason.blend_forms(prior_forms,
                                          preseason.live_form(before_lines), played)

            for g in (x for x in season_games if x.week == week):
                home_form, away_form = forms.get(g.home), forms.get(g.away)
                if not (home_form and away_form
                        and home_form.complete() and away_form.complete()):
                    continue
                base = ratings.projected_margin(table, g.home, g.away, neutral=g.neutral)
                if base is None:
                    continue
                market = next((r for r in schedule
                               if r["season"] == season and r["week"] == week
                               and teams.canonical(r["home_team"]) == g.home
                               and teams.canonical(r["away_team"]) == g.away), None)
                rows.append({
                    "season": season, "week": week, "home": g.home, "away": g.away,
                    "home_points": g.home_points, "away_points": g.away_points,
                    "margin": g.margin, "total": g.home_points + g.away_points,
                    "base": base, "neutral": g.neutral,
                    "market_margin": (market or {}).get("spread_line"),
                    "market_total": (market or {}).get("total_line"),
                    "home_form": home_form, "away_form": away_form,
                })
    return rows


# -- fitting ------------------------------------------------------------------
def side(offense, defence, home: bool) -> list[float]:
    """Feature row for one scoring side."""
    return ([getattr(offense, f"off_{s}") + getattr(defence, f"def_{s}") for s in STATS]
            + [1.0 if home else 0.0])


COLUMNS = list(STATS) + ["home_field"]


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(beta[0]), beta[1:]


def side_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, season = [], [], []
    for row in rows:
        X.append(side(row["home_form"], row["away_form"], not row["neutral"]))
        y.append(row["home_points"])
        season.append(row["season"])
        X.append(side(row["away_form"], row["home_form"], False))
        y.append(row["away_points"])
        season.append(row["season"])
    return np.array(X, float), np.array(y, float), np.array(season)


def leave_one_season_out(rows: list[dict]) -> tuple[list[float], list[float]]:
    """Point projections for both sides of every game, from a model that never
    saw that game's season."""
    X, y, seasons = side_matrix(rows)
    home_points: dict[int, float] = {}
    away_points: dict[int, float] = {}
    for season in sorted(set(seasons)):
        intercept, beta = _ols(X[seasons != season], y[seasons != season])
        for index, row in enumerate(rows):
            if row["season"] != season:
                continue
            home_points[index] = intercept + float(
                np.array(side(row["home_form"], row["away_form"], not row["neutral"])) @ beta)
            away_points[index] = intercept + float(
                np.array(side(row["away_form"], row["home_form"], False)) @ beta)
    return ([home_points[i] for i in range(len(rows))],
            [away_points[i] for i in range(len(rows))])


def mae(actual, predicted) -> float:
    return float(np.mean(np.abs(np.array(actual, float) - np.array(predicted, float))))


def record(rows, predicted, market_key, actual_key, over_under=False):
    """(wins, losses, pushes, rate, standard error) against the market."""
    wins = losses = pushes = 0
    for row, prediction in zip(rows, predicted):
        market = row[market_key]
        if market is None or abs(prediction - market) < 1e-9:
            continue
        edge = row[actual_key] - market
        if abs(edge) < 1e-9:
            pushes += 1
        elif (edge > 0) == (prediction > market):
            wins += 1
        else:
            losses += 1
    played = wins + losses
    rate = wins / played if played else 0.0
    se = (rate * (1 - rate) / played) ** 0.5 if played else 0.0
    return wins, losses, pushes, rate, se


# -- sweeps -------------------------------------------------------------------
def sweep_context(schedule: list[dict]) -> dict:
    history = ratings.from_rows(schedule)
    hosted = [g for g in history if not g.neutral]
    early = [g for g in hosted if 2014 <= g.season <= 2019]
    modern = [g for g in hosted if g.season >= 2021]
    blowouts = sum(1 for g in history if abs(g.margin) >= 28)
    out = {
        "home_margin_2014_2019": statistics.fmean(g.margin for g in early),
        "home_margin_2021_2025": statistics.fmean(g.margin for g in modern),
        "blowout_share": blowouts / len(history),
        "margin_sd": statistics.stdev(g.margin for g in history),
        "total_sd": statistics.stdev(g.home_points + g.away_points for g in history),
        "total_mean": statistics.fmean(g.home_points + g.away_points for g in history),
    }
    print("\n  measured context")
    print(f"    mean home margin 2014-2019   {out['home_margin_2014_2019']:+.3f}")
    print(f"    mean home margin 2021-2025   {out['home_margin_2021_2025']:+.3f}")
    print(f"    games decided by 28+          {out['blowout_share']:.1%} "
          f"(cfb-model measures 36% in FBS)")
    print(f"    margin SD {out['margin_sd']:.2f}   total SD {out['total_sd']:.2f}   "
          f"total mean {out['total_mean']:.2f}")
    return out


def sweep_halflife(schedule, lines) -> None:
    print("\n  in-season recency half-life (blended margin MAE)")
    original = ratings.RECENCY_HALFLIFE_WEEKS
    for halflife in (4.0, 8.0, 12.0, 20.0, 40.0, None):
        ratings.RECENCY_HALFLIFE_WEEKS = halflife
        rows = build_features(schedule, lines)
        home, away = leave_one_season_out(rows)
        blended = [totals.EFFICIENCY_WEIGHT * (h - a)
                   + (1 - totals.EFFICIENCY_WEIGHT) * r["base"]
                   for r, h, a in zip(rows, home, away)]
        label = "none" if halflife is None else f"{halflife:.0f}"
        print(f"    half-life {label:>5}  MAE {mae([r['margin'] for r in rows], blended):.4f}")
    ratings.RECENCY_HALFLIFE_WEEKS = original


# -- report -------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-sweeps", action="store_true")
    parser.add_argument("--out", default="reports/fit_summary.json")
    args = parser.parse_args()

    print("loading nflverse data")
    schedule, lines = load()
    context = sweep_context(schedule)
    if not args.no_sweeps:
        sweep_halflife(schedule, lines)

    print("\nbuilding point-in-time features")
    rows = build_features(schedule, lines)
    print(f"  {len(rows)} usable games {FIRST_TARGET}-{LAST_TARGET}")

    home_points, away_points = leave_one_season_out(rows)
    efficiency_margin = [h - a for h, a in zip(home_points, away_points)]
    raw_total = [h + a for h, a in zip(home_points, away_points)]
    league_total = statistics.fmean(r["total"] for r in rows)
    projected_total = [league_total + totals.TOTAL_SHRINK * (t - league_total)
                       for t in raw_total]
    base = [r["base"] for r in rows]
    w = totals.EFFICIENCY_WEIGHT
    blended = [w * e + (1 - w) * b for e, b in zip(efficiency_margin, base)]

    actual_margin = [r["margin"] for r in rows]
    actual_total = [r["total"] for r in rows]
    priced = [r for r in rows if r["market_margin"] is not None]
    priced_total = [r for r in rows if r["market_total"] is not None]

    print("\n== MARGIN ==================================================")
    print(f"  ratings-only        {mae(actual_margin, base):.4f}")
    print(f"  efficiency-only     {mae(actual_margin, efficiency_margin):.4f}")
    print(f"  blended (w={w:.2f})     {mae(actual_margin, blended):.4f}")
    print(f"  market              "
          f"{mae([r['margin'] for r in priced], [r['market_margin'] for r in priced]):.4f}"
          f"   on {len(priced)} priced games")
    wins, losses, pushes, rate, se = record(rows, blended, "market_margin", "margin")
    print(f"  ATS on disagreements  {wins}-{losses}-{pushes} = {rate:.2%} "
          f"(95% CI [{rate - 1.96 * se:.2%}, {rate + 1.96 * se:.2%}], breakeven 52.38%)")

    print("\n== TOTAL ===================================================")
    print(f"  league mean ({league_total:.2f})  "
          f"{mae(actual_total, [league_total] * len(rows)):.4f}")
    print(f"  unshrunk model      {mae(actual_total, raw_total):.4f}")
    print(f"  shrunk (lam={totals.TOTAL_SHRINK:.2f})    {mae(actual_total, projected_total):.4f}")
    market_total_mae = mae([r["total"] for r in priced_total],
                           [r["market_total"] for r in priced_total])
    print(f"  market              {market_total_mae:.4f}"
          f"   on {len(priced_total)} priced games")
    ow, ol, op, orate, ose = record(rows, projected_total, "market_total", "total")
    print(f"  O/U on disagreements  {ow}-{ol}-{op} = {orate:.2%} "
          f"(95% CI [{orate - 1.96 * ose:.2%}, {orate + 1.96 * ose:.2%}])")

    print("\n== BY WEEK =================================================")
    buckets = []
    for lo, hi, label in ((1, 4, "weeks 1-4"), (5, 9, "weeks 5-9"), (10, 18, "weeks 10-18")):
        idx = [i for i, r in enumerate(rows)
               if lo <= r["week"] <= hi and r["market_margin"] is not None]
        model = mae([rows[i]["margin"] for i in idx], [blended[i] for i in idx])
        market = mae([rows[i]["margin"] for i in idx],
                     [rows[i]["market_margin"] for i in idx])
        dispersion = (statistics.stdev(blended[i] for i in idx)
                      / statistics.stdev(rows[i]["market_margin"] for i in idx))
        tidx = [i for i, r in enumerate(rows)
                if lo <= r["week"] <= hi and r["market_total"] is not None]
        tmodel = mae([rows[i]["total"] for i in tidx], [projected_total[i] for i in tidx])
        tmarket = mae([rows[i]["total"] for i in tidx],
                      [rows[i]["market_total"] for i in tidx])
        print(f"  {label:<12} n={len(idx):4d}  margin {model:.4f} vs {market:.4f} "
              f"({model - market:+.4f})  dispersion {dispersion:.3f}")
        print(f"  {'':<12}            total  {tmodel:.4f} vs {tmarket:.4f} "
              f"({tmodel - tmarket:+.4f})")
        buckets.append({"label": label, "n": len(idx), "model": model, "market": market,
                        "dispersion": dispersion, "total_model": tmodel,
                        "total_market": tmarket})

    # Full-sample coefficients: what ships.
    X, y, _ = side_matrix(rows)
    intercept, beta = _ols(X, y)
    coefficients = {"intercept": intercept}
    coefficients.update({name: float(value) for name, value in zip(COLUMNS, beta)})

    league_form = {s: statistics.fmean(
        [getattr(r["home_form"], f"off_{s}") for r in rows]
        + [getattr(r["away_form"], f"off_{s}") for r in rows]) for s in STATS}

    # Standardised importance. Offence and defence share a coefficient but NOT a
    # spread: team-to-team variation in offensive EPA is larger than in defensive
    # EPA allowed, so the two groups weight differently and both are worth showing.
    def weights(prefix: str) -> dict[str, float]:
        raw = {}
        for stat in STATS:
            values = ([getattr(r["home_form"], f"{prefix}_{stat}") for r in rows]
                      + [getattr(r["away_form"], f"{prefix}_{stat}") for r in rows])
            raw[stat] = abs(coefficients[stat]) * float(np.std(values))
        total = sum(raw.values())
        return {k: round(v / total, 4) for k, v in raw.items()}

    offense_weights = weights("off")
    defense_weights = weights("def")

    print("\n== COEFFICIENTS (points scored by one side) ================")
    for key, value in coefficients.items():
        print(f"  {key:<14} {value:>12.5f}")
    print("\n  offence weights (standardised, normalised)")
    for key, value in sorted(offense_weights.items(), key=lambda kv: -kv[1]):
        print(f"    {key:<14} {value:.4f}")
    print("  defence weights")
    for key, value in sorted(defense_weights.items(), key=lambda kv: -kv[1]):
        print(f"    {key:<14} {value:.4f}")
    print("\n  league mean adjusted rate")
    for key, value in league_form.items():
        print(f"    {key:<14} {value:>12.6f}")

    margin_sd = statistics.stdev(a - p for a, p in zip(actual_margin, blended))
    total_sd = statistics.stdev(a - p for a, p in zip(actual_total, projected_total))
    slope_margin = np.linalg.lstsq(
        np.column_stack([np.ones(len(rows)), blended]),
        np.array(actual_margin), rcond=None)[0]
    slope_total = np.linalg.lstsq(
        np.column_stack([np.ones(len(rows)), projected_total]),
        np.array(actual_total), rcond=None)[0]
    print(f"\n  margin residual SD {margin_sd:.4f}   calibration slope "
          f"{slope_margin[1]:.4f}")
    print(f"  total  residual SD {total_sd:.4f}   calibration slope "
          f"{slope_total[1]:.4f}")

    payload = {
        "games": len(rows),
        "seasons": [FIRST_TARGET, LAST_TARGET],
        "context": context,
        "coefficients": coefficients,
        "league_mean_form": league_form,
        "offense_weights": offense_weights,
        "defense_weights": defense_weights,
        "margin": {
            "ratings_only_mae": mae(actual_margin, base),
            "efficiency_only_mae": mae(actual_margin, efficiency_margin),
            "model_mae": mae(actual_margin, blended),
            "market_mae": mae([r["margin"] for r in priced],
                              [r["market_margin"] for r in priced]),
            "ats": [wins, losses, pushes],
            "ats_rate": rate,
            "residual_sd": margin_sd,
            "calibration_slope": float(slope_margin[1]),
        },
        "total": {
            "league_mean": league_total,
            "league_mean_mae": mae(actual_total, [league_total] * len(rows)),
            "model_mae": mae(actual_total, projected_total),
            "market_mae": mae([r["total"] for r in priced_total],
                              [r["market_total"] for r in priced_total]),
            "ou": [ow, ol, op],
            "ou_rate": orate,
            "residual_sd": total_sd,
            "calibration_slope": float(slope_total[1]),
        },
        "buckets": buckets,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
