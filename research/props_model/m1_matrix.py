"""Player prop matrix: log-link (Poisson) regression on a structural baseline.

    log E[stat] = log(baseline) + sum_k beta_k * x_k

baseline = team volume forecast x usage share (volume stats), or volume
baseline x efficiency prior (yardage). Each x_k is a pre-game, schematically
motivated factor (usage regression, snap trend, opponent position-group
defense, pressure matchup, coverage interaction, box rate, game script).
Fitted on earlier seasons, scored on the next (2024, 2025).
"""
import sys, json
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, PoissonRegressor

f = pd.read_parquet("matrix_features.parquet")
tg = pd.read_parquet("team_games_feat.parquet")
f = f[f.season.between(2022, 2025) & (f.p_games >= 1)].copy()
f["grp"] = f.position.replace({"FB": "RB"})
f = f[f.grp.isin(["WR", "TE", "RB"])]

POS_T = f.groupby("grp").tshare.mean().to_dict()
POS_C = f.groupby("grp").cshare.mean().to_dict()
LG = {"adot": {"WR": 10.66, "TE": 6.8, "RB": 0.1}}
YPT_A = {"WR": (6.08, 0.170), "TE": (5.633, 0.273), "RB": (5.824, 0.228)}
CR_A = {"WR": (0.807, -0.0172), "TE": (0.81, -0.0145), "RB": (0.787, -0.0163)}

def team_hats(train_seasons):
    d = tg[tg.season.isin(train_seasons) & (tg.week >= 2)].dropna(subset=["o_pass", "d_pass", "o_rush", "d_rush", "t_spread", "total_line"])
    X = ["o_pass", "d_pass", "t_spread", "total_line"]; XR = ["o_rush", "d_rush", "t_spread", "total_line"]
    mp = LinearRegression().fit(d[X], d.pass_att); mr = LinearRegression().fit(d[XR], d.designed_rush)
    tr = (d.pass_att.sum() and (tg[tg.season.isin(train_seasons)].pass_att.sum()))
    return mp, mr, X, XR

def build(d, mp, mr, X, XR):
    d = d.copy()
    ok = d[X].notna().all(axis=1) & d[XR].notna().all(axis=1)
    d = d[ok]
    d["pass_hat"] = mp.predict(d[X]); d["rush_hat"] = mr.predict(d[XR])
    d["tt_hat"] = d.pass_hat * 0.93  # targets per attempt (sacks excluded; throwaways)
    d["share0"] = d.p_tshare.fillna(d.grp.map(POS_T)).clip(lower=0.002)
    d["cshare0"] = d.p_cshare.fillna(d.grp.map(POS_C)).clip(lower=0.002)
    d["tgt0"] = d.tt_hat * d.share0
    d["car0"] = d.rush_hat * d.cshare0
    adot = (d.s_air.fillna(0) * 1 + 2.0 * d.grp.map(LG["adot"])) / (d.s_targets.fillna(0) + 2.0)
    d["adot"] = adot
    ypt_pr = np.array([YPT_A[g][0] + YPT_A[g][1] * a for g, a in zip(d.grp, adot)])
    cr_pr = np.array([CR_A[g][0] + CR_A[g][1] * a for g, a in zip(d.grp, adot)])
    # s_* are EW per-game means; ratio of means = rate. shrink with ~4 games of targets
    d["ypt0"] = (d.s_rec_yds.fillna(0) + 4 * 4 * ypt_pr / 4 * 1) / (d.s_targets.fillna(0) + 4)  # placeholder overwritten below
    k_t = 38 / 6.0  # 38 targets of pseudo-count expressed in per-game EW units (~6 targets/game)
    d["ypt0"] = (d.s_rec_yds.fillna(0) + k_t * ypt_pr) / (d.s_targets.fillna(0) + k_t)
    d["cr0"] = (d.s_receptions.fillna(0) + k_t * cr_pr) / (d.s_targets.fillna(0) + k_t)
    k_c = 45 / 12.0
    d["ypc0"] = (d.s_rush_yds.fillna(0) + k_c * 4.3) / (d.s_carries.fillna(0) + k_c)
    d["ryd0"] = d.tgt0 * d.ypt0
    d["rec0"] = d.tgt0 * d.cr0
    d["rush0"] = d.car0 * d.ypc0
    # ---- factors
    g = d.grp
    d["x_share_reg"] = np.log(d.share0 / g.map(POS_T))          # >0 heavy-usage player; beta<0 = regress to mean
    d["x_cshare_reg"] = np.log(d.cshare0 / g.map(POS_C))
    d["x_snap_trend"] = np.log((d.p_last_snap.fillna(d.p_snap) + 0.02) / (d.p_snap.fillna(0.5) + 0.02)).clip(-2, 2)
    d["x_low_games"] = 1.0 / (1.0 + d.p_games)
    sh = {"WR": "d_share_WR", "TE": "d_share_TE", "RB": "d_share_RB"}
    yp = {"WR": "d_ypt_WR", "TE": "d_ypt_TE", "RB": "d_ypt_RB"}
    cr = {"WR": "d_cr_WR", "TE": "d_cr_TE", "RB": "d_cr_RB"}
    d["x_def_share"] = [np.log(r[sh[gg]] / r[sh[gg] + "_lg"]) if r[sh[gg] + "_lg"] > 0 else 0 for gg, (_, r) in zip(g, d.iterrows())]
    d["x_def_ypt"] = [np.log(r[yp[gg]] / r[yp[gg] + "_lg"]) for gg, (_, r) in zip(g, d.iterrows())]
    d["x_def_cr"] = [np.log(r[cr[gg]] / r[cr[gg] + "_lg"]) for gg, (_, r) in zip(g, d.iterrows())]
    d["x_def_pepa"] = d.d_pepa - d.d_pepa_lg
    d["x_pressure"] = (d.o_sack - d.o_sack_lg) + (d.d_sack - d.d_sack_lg)     # expected pressure in this matchup
    d["x_press_adot"] = d.x_pressure * (d.adot - 7.0) / 5.0                  # pressure hurts deep roles
    d["x_press_rb"] = d.x_pressure * (g == "RB")                             # and feeds check-downs
    d["x_cov"] = d.man_tilt.fillna(0) * d.opp_man_dev.fillna(0) * 10          # man-beater vs man-heavy defense
    d["x_blitz_rb"] = (d.d_blitz - d.d_blitz_lg) * (g == "RB") * 10
    d["x_blitz_te"] = (d.d_blitz - d.d_blitz_lg) * (g == "TE") * 10
    d["x_box"] = (d.d_box - d.d_box_lg) * 10
    d["x_def_ypc"] = np.log(d.d_ypc / d.d_ypc_lg)
    d["x_off_ypc"] = np.log(d.o_ypc / d.o_ypc_lg)
    d["x_spread_rb"] = d.t_spread * (g == "RB") / 7
    d["x_spread"] = d.t_spread / 7
    d["x_total"] = (d.total_line - 44.5) / 7
    d["x_off_pepa"] = d.o_pepa - d.o_pepa_lg
    xs = [c for c in d.columns if c.startswith("x_")]
    d[xs] = d[xs].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for c in ("tgt0", "car0", "ryd0", "rec0", "rush0"):
        d[c] = d[c].astype(float).fillna(d[c].median())
    return d

FAMILIES = {
    "usage": ["x_share_reg", "x_snap_trend", "x_low_games"],
    "opponent": ["x_def_share", "x_def_ypt", "x_def_cr", "x_def_pepa"],
    "pressure": ["x_pressure", "x_press_adot", "x_press_rb"],
    "coverage": ["x_cov"],
    "blitz_box": ["x_blitz_rb", "x_blitz_te"],
    "script": ["x_spread", "x_spread_rb", "x_total", "x_off_pepa"],
}
RUSH_FAMILIES = {
    "usage": ["x_cshare_reg", "x_snap_trend", "x_low_games"],
    "opponent": ["x_def_ypc", "x_off_ypc"],
    "box": ["x_box"],
    "script": ["x_spread", "x_total"],
}
TARGETS = {
    "targets": ("targets", "tgt0", FAMILIES, ["WR", "TE", "RB"]),
    "receptions": ("receptions", "rec0", FAMILIES, ["WR", "TE", "RB"]),
    "receiving_yards": ("rec_yds", "ryd0", FAMILIES, ["WR", "TE", "RB"]),
    "carries": ("carries", "car0", RUSH_FAMILIES, ["RB"]),
    "rushing_yards": ("rush_yds", "rush0", RUSH_FAMILIES, ["RB"]),
}

def fit(d, y, base, feats, alpha=1e-3):
    w = d[base].clip(lower=1e-3)
    m = PoissonRegressor(alpha=alpha, max_iter=1000).fit(d[feats], d[y].clip(lower=0) / w, sample_weight=w) if feats else None
    return m

def predict(m, d, base, feats):
    return d[base].clip(lower=1e-3) * (m.predict(d[feats]) if m is not None else 1.0)

def score(y, mu):
    err = mu - y
    dev = 2 * np.where(y > 0, y * np.log(np.maximum(y, 1e-9) / mu), 0) - 2 * (y - mu)
    return np.sqrt(np.mean(err ** 2)), np.mean(err), np.mean(dev)

results = {}
coefs = {}
for name, (y, base, fams, groups) in TARGETS.items():
    rows = []
    for test in (2024, 2025):
        train = list(range(2022, test))
        mp, mr, X, XR = team_hats(train)
        tr = build(f[f.season.isin(train) & f.grp.isin(groups)], mp, mr, X, XR)
        te = build(f[(f.season == test) & f.grp.isin(groups)], mp, mr, X, XR)
        tr = tr[tr[y].notna()]; te = te[te[y].notna()]
        line = {"test": test, "n": len(te)}
        rmse0, b0, d0 = score(te[y].values, te[base].values.clip(1e-3))
        line["baseline"] = (round(rmse0, 3), round(b0, 3), round(d0, 4))
        # intercept-only recalibration of the baseline
        cum = []
        for fam, cols in fams.items():
            cum = cum + cols
            m = fit(tr, y, base, cum)
            r, b, dv = score(te[y].values, predict(m, te, base, cum))
            line["+" + fam] = (round(r, 3), round(b, 3), round(dv, 4))
        # drop-one-family ablation from the full set
        full = sum(fams.values(), [])
        for fam, cols in fams.items():
            keep = [c for c in full if c not in cols]
            m = fit(tr, y, base, keep)
            r, b, dv = score(te[y].values, predict(m, te, base, keep))
            line["-" + fam] = (round(r, 3), round(b, 3), round(dv, 4))
        rows.append(line)
        if test == 2025:
            m = fit(tr, y, base, full)
            coefs[name] = dict(zip(["intercept"] + full, [round(float(m.intercept_), 4)] + [round(float(c), 4) for c in m.coef_]))
    results[name] = rows
    print(f"\n=== {name}")
    for line in rows:
        print(f"  test {line['test']} n={line['n']}")
        for k, v in line.items():
            if k in ("test", "n"): continue
            print(f"     {k:14s} rmse {v[0]:8.3f}  bias {v[1]:+7.3f}  deviance {v[2]:.4f}")
json.dump({"results": results, "coefs_fit_2022_2024": coefs}, open("matrix_results.json", "w"), indent=1)
print(json.dumps(coefs, indent=1))
