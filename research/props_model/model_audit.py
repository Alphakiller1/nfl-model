"""Time-forward: spread skill vs market; QB-out; skill-position absences on team points."""
import sys, statistics
from collections import defaultdict
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import numpy as np
import audit_regimes as audit, fit_matrix as fit, fit_availability as fa
from nflmodel import totals, availability
from nflmodel.sources import nflverse
schedule, lines = fit.load()
rows = fit.build_features(schedule, lines)
games = []
for ts in sorted({int(r["season"]) for r in rows}):
    if ts < audit.FIRST_TEST_SEASON: continue
    tr = [r for r in rows if r["season"] < ts]; te = [r for r in rows if r["season"] == ts]
    X, Y, _ = fit.side_matrix(tr); a, beta = fit._ols(X, Y)
    for r in te:
        h = a + float(np.asarray(fit.side(r["home_form"], r["away_form"], not r["neutral"])) @ beta)
        w = a + float(np.asarray(fit.side(r["away_form"], r["home_form"], False)) @ beta)
        games.append({"season": ts, "week": int(r["week"]), "home": r["home"], "away": r["away"],
                      "actual": float(r["margin"]), "market": r["market_margin"],
                      "model": 0.5 * float(r["base"]) + 0.5 * (h - w), "h_pts": h, "a_pts": w,
                      "total": float(r["total"]), "market_total": r.get("market_total")})
fa.flag_games(games)
for g in games:
    g["model_qb"] = g["model"] + availability.QB_OUT_POINTS * g["flag"]
# ---- 1. spread skill
pr = [g for g in games if g["market"] is not None]
def w_of(xs, key):
    num = sum((g[key] - g["market"]) * (g["actual"] - g["market"]) for g in xs)
    den = sum((g[key] - g["market"]) ** 2 for g in xs); return num / den
print("SPREAD: model margin blend weight vs market (time-forward model, QB adj):")
for s in sorted({g["season"] for g in pr}):
    print(f"  {s}: w {w_of([g for g in pr if g['season'] == s], 'model_qb'):+.3f}")
print(f"  all: w {w_of(pr, 'model_qb'):+.3f}")
for lo in (2, 4, 6):
    xs = [g for g in pr if abs(g["model_qb"] - g["market"]) >= lo and g["actual"] != g["market"]]
    hit = sum(((g["model_qb"] > g["market"]) == (g["actual"] > g["market"])) for g in xs)
    print(f"  ATS when |model - market| >= {lo}: {hit}-{len(xs) - hit} ({hit / len(xs):.1%})")
mae = lambda xs: statistics.fmean(abs(x) for x in xs)
print(f"  MAE margin: model {mae(g['actual'] - g['model_qb'] for g in pr):.3f} market {mae(g['actual'] - g['market'] for g in pr):.3f}")
# ---- 2. skill-position absences: top target earner / lead back unavailable
def flag_skill(games):
    for season in sorted({g["season"] for g in games}):
        players = fa._season_rows(nflverse.PLAYER_WEEK_URL.format(season=season), f"player_stats_week_{season}.csv")
        inj = fa._by_week(fa._season_rows(nflverse.INJURIES_URL.format(season=season), f"injuries_{season}.csv"))
        ros = fa._by_week(fa._season_rows(nflverse.WEEKLY_ROSTER_URL.format(season=season), f"roster_weekly_{season}.csv"))
        bt = defaultdict(lambda: defaultdict(list))
        for r in players:
            if r.get("season_type") == "REG": bt[r["team"]][int(r["week"])].append(r)
        for week in sorted({g["week"] for g in games if g["season"] == season}):
            out = {(r.get("team"), r.get("gsis_id")) for r in inj.get(week, []) if str(r.get("report_status", "")).lower() in ("out", "doubtful")}
            out |= {(r.get("team"), r.get("gsis_id")) for r in ros.get(week, []) if r.get("status") in availability.UNAVAILABLE_ROSTER}
            flags = {}
            for team, wk in bt.items():
                recent = [wk[w] for w in sorted(wk) if w < week][-4:]
                if len(recent) < 2: continue
                tshare, cshare = defaultdict(float), defaultdict(float)
                for rs in recent:
                    tt = sum(float(r["targets"] or 0) for r in rs) or 1; cc = sum(float(r["carries"] or 0) for r in rs) or 1
                    for r in rs:
                        tshare[r["player_id"]] += float(r["targets"] or 0) / tt / len(recent)
                        if r.get("position") == "RB": cshare[r["player_id"]] += float(r["carries"] or 0) / cc / len(recent)
                top_t = max(tshare, key=tshare.get); top_c = max(cshare, key=cshare.get) if cshare else None
                flags[team] = (int(tshare[top_t] >= 0.20 and (team, top_t) in out),
                               int(top_c is not None and cshare[top_c] >= 0.45 and (team, top_c) in out))
            for g in games:
                if g["season"] == season and g["week"] == week:
                    g["h_wr"], g["h_rb"] = flags.get(g["home"], (0, 0)); g["a_wr"], g["a_rb"] = flags.get(g["away"], (0, 0))
flag_skill(games)
# team-perspective rows: residual of team points
tp = []
for g in games:
    hp = (g["total"] + g["actual"]) / 2; ap = (g["total"] - g["actual"]) / 2
    tp.append({"res": hp - g["h_pts"], "wr": g.get("h_wr", 0), "rb": g.get("h_rb", 0), "qb": g["home_out"], "season": g["season"]})
    tp.append({"res": ap - g["a_pts"], "wr": g.get("a_wr", 0), "rb": g.get("a_rb", 0), "qb": g["away_out"], "season": g["season"]})
X = np.array([[1, t["qb"], t["wr"], t["rb"]] for t in tp], float); y = np.array([t["res"] for t in tp])
coef, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ coef
cov = np.linalg.inv(X.T @ X) * (res @ res) / (len(y) - 4); se = np.sqrt(np.diag(cov))
print(f"\nTEAM POINTS residual vs absences (n={len(tp)} team-games; qb {int(X[:,1].sum())}, top receiver {int(X[:,2].sum())}, lead back {int(X[:,3].sum())} flagged):")
for name, c, s in zip(("intercept", "QB out", "top target earner out", "lead back out"), coef, se):
    print(f"  {name:22s} {c:+.2f} pts (se {s:.2f}, t {c / s:+.1f})")
