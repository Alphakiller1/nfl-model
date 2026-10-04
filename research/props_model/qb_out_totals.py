"""Does a missing starting QB lower the game total, and does the model know? Time-forward."""
import sys, statistics
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import numpy as np
import audit_regimes as audit, fit_matrix as fit, fit_availability as fa
from nflmodel import totals
schedule, lines = fit.load()
rows = fit.build_features(schedule, lines)
games = []
for test_season in sorted({int(r["season"]) for r in rows}):
    if test_season < audit.FIRST_TEST_SEASON: continue
    train = [r for r in rows if r["season"] < test_season]; test = [r for r in rows if r["season"] == test_season]
    X, Y, _ = fit.side_matrix(train); a, beta = fit._ols(X, Y)
    for r in test:
        h = a + float(np.asarray(fit.side(r["home_form"], r["away_form"], not r["neutral"])) @ beta)
        w = a + float(np.asarray(fit.side(r["away_form"], r["home_form"], False)) @ beta)
        games.append({"season": test_season, "week": int(r["week"]), "home": r["home"], "away": r["away"],
                      "actual": float(r["margin"]), "market": r["market_margin"], "model": 0.0,
                      "total": float(r["total"]), "market_total": r.get("market_total"),
                      "model_total": totals.shrink_total(h + w)})
fa.flag_games(games)
for g in games: g["outs"] = g["home_out"] + g["away_out"]
def slope(gs, f):
    x = np.array([g["outs"] for g in gs], float); y = np.array([f(g) for g in gs], float)
    # regress residual on outs with intercept (league-wide model bias absorbed)
    A = np.c_[np.ones_like(x), x]; coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef; cov = np.linalg.inv(A.T @ A) * (res @ res) / (len(x) - 2)
    return coef[1], np.sqrt(cov[1, 1])
b, se = slope(games, lambda g: g["total"] - g["model_total"])
priced = [g for g in games if g["market_total"] is not None]
mb, mse = slope(priced, lambda g: float(g["market_total"]) - g["model_total"])
print(f"games {len(games)}, with a starter out {sum(g['outs']>0 for g in games)}")
print(f"actual total - model total per QB out: {b:+.2f} (se {se:.2f})")
print(f"market total - model total per QB out: {mb:+.2f} (se {mse:.2f})")
# time-forward: coefficient fitted on earlier seasons, scored on the next
sc = []
for s in sorted({g["season"] for g in games}):
    tr = [g for g in games if g["season"] < s]
    if len(tr) < 200: continue
    bb, _ = slope(tr, lambda g: g["total"] - g["model_total"])
    for g in games:
        if g["season"] == s: sc.append((g, g["model_total"] + bb * g["outs"]))
fl = [(g, adj) for g, adj in sc if g["outs"] > 0]
mae = lambda xs: statistics.fmean(abs(x) for x in xs)
print(f"flagged games scored time-forward: {len(fl)}")
print(f"  MAE total: model {mae(g['total'] - g['model_total'] for g, _ in fl):.3f} -> with QB-out total adj {mae(g['total'] - a for g, a in fl):.3f}; market {mae(g['total'] - float(g['market_total']) for g, _ in fl if g['market_total'] is not None):.3f}")
print(f"  all games: model {mae(g['total'] - g['model_total'] for g, _ in sc):.3f} -> {mae(g['total'] - a for g, a in sc):.3f}")

# How much of (model - market) shows up in (actual - market)? time-forward, with the QB-out total adj
print("\nblend weight of the model total against the market total:")
pr = [(g, a) for g, a in sc if g["market_total"] is not None]
def w_of(xs):
    num = sum((a - float(g["market_total"])) * (g["total"] - float(g["market_total"])) for g, a in xs)
    den = sum((a - float(g["market_total"])) ** 2 for g, a in xs)
    return num / den
for s in sorted({g["season"] for g, _ in pr}):
    print(f"  {s}: w {w_of([x for x in pr if x[0]['season'] == s]):+.3f}")
w_all = w_of(pr); print(f"  all: w {w_all:+.3f}")
big = [(g, a) for g, a in pr if abs(a - float(g["market_total"])) >= 3]
hit = sum((g["total"] > float(g["market_total"])) == (a > float(g["market_total"])) for g, a in big if g["total"] != float(g["market_total"]))
n = sum(1 for g, a in big if g["total"] != float(g["market_total"]))
print(f"  side record when |model - market| >= 3: {hit}-{n - hit} ({hit / n:.1%})")
big6 = [(g, a) for g, a in pr if abs(a - float(g["market_total"])) >= 6 and g["total"] != float(g["market_total"])]
h6 = sum((g["total"] > float(g["market_total"])) == (a > float(g["market_total"])) for g, a in big6)
print(f"  side record when |model - market| >= 6: {h6}-{len(big6) - h6}")
