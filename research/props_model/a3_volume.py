"""Pre-game team volume model: pass attempts & designed rushes, time-forward validated."""
import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression, Ridge

t = pd.read_parquet("team_games.parquet").sort_values(["season", "week"]).reset_index(drop=True)
t["db_rate"] = t.dropbacks / t.plays

def pit(metric, key, h, kp, kc, out):
    lg = t[t.season < 2026].groupby("season")[metric].mean()
    lg = lg.reindex(range(2021, 2027)).ffill()
    last = t.groupby(["season", key])[metric].mean()
    vals = np.full(len(t), np.nan)
    for (s, team), d in t.groupby(["season", key]):
        base = lg[s - 1] if s > 2021 else lg[s]
        prior = base + kc * (last.get((s - 1, team), base) - base) if s > 2021 else base
        xs = d[metric].values; wk = d.week.values
        for j, idx in enumerate(d.index):
            w = 0.5 ** ((wk[j] - wk[:j]) / h); v = xs[:j]; ok = ~np.isnan(v)
            vals[idx] = (np.sum(w[ok] * v[ok]) + kp * prior) / (np.sum(w[ok]) + kp)
    t[out] = vals

# Optimal point-in-time priors (A2)
pit("pass_att", "posteam", 4, 4, 0.5, "o_pass")
pit("designed_rush", "posteam", 4, 6, 0.3, "o_rush")
pit("n_proe", "posteam", 4, 4, 0.3, "o_proe")
pit("n_sec", "posteam", 8, 4, 0.3, "o_sec")
pit("plays", "posteam", 4, 10, 0.3, "o_plays")
pit("pass_att", "defteam", 4, 10, 0.3, "d_pass")
pit("designed_rush", "defteam", 4, 10, 0.3, "d_rush")
pit("plays", "defteam", 4, 20, 0.0, "d_plays")
pit("scrambles", "posteam", 4, 6, 0.5, "o_scr")
# opponent's own offensive pace & PROE (their possessions eat clock)
opp = t[["game_id", "posteam", "o_sec", "o_proe", "o_plays"]].rename(
    columns={"posteam": "defteam", "o_sec": "opp_sec", "o_proe": "opp_proe", "o_plays": "opp_plays"})
t = t.merge(opp, on=["game_id", "defteam"], how="left")

# Production-style baseline (12-wk half-life, pseudo 6/8, 58/42, -0.16 margin)
pit("pass_att", "posteam", 12, 6, 1.0, "p_team"); pit("pass_att", "defteam", 12, 8, 1.0, "p_allow")
pit("designed_rush", "posteam", 12, 6, 1.0, "pr_team"); pit("designed_rush", "defteam", 12, 8, 1.0, "pr_allow")
t["prod_pass"] = (0.58 * t.p_team + 0.42 * t.p_allow - 0.16 * t.t_spread).clip(25, 44)
t["prod_rush"] = (0.58 * t.pr_team + 0.42 * t.pr_allow + 0.13 * t.t_spread).clip(19, 35)
t.to_parquet("team_games_feat.parquet")

FEATS = {
    "market only": ["t_spread", "total_line"],
    "priors only": ["o_pass", "d_pass", "o_proe", "o_sec", "opp_sec"],
    "priors + market": ["o_pass", "d_pass", "o_proe", "o_sec", "opp_sec", "t_spread", "total_line"],
}
RFEATS = {
    "market only": ["t_spread", "total_line"],
    "priors only": ["o_rush", "d_rush", "o_proe", "o_sec", "opp_sec"],
    "priors + market": ["o_rush", "d_rush", "o_proe", "o_sec", "opp_sec", "t_spread", "total_line"],
}
d = t[t.week >= 2].dropna(subset=["o_pass", "d_pass", "o_proe", "o_sec", "opp_sec", "t_spread", "total_line"])
for target, feats, prod in (("pass_att", FEATS, "prod_pass"), ("designed_rush", RFEATS, "prod_rush")):
    print(f"\n=== {target}")
    for test in (2024, 2025):
        tr = d[(d.season >= 2022) & (d.season < test)]; te = d[d.season == test]
        sd = te[target].std()
        line = f"test {test} (sd {sd:.2f}): production MAE {np.mean(np.abs(te[prod]-te[target])):.3f}"
        for name, f in feats.items():
            m = LinearRegression().fit(tr[f], tr[target]); pr = m.predict(te[f])
            line += f" | {name} {np.mean(np.abs(pr-te[target])):.3f}"
        print(line)
    m = LinearRegression().fit(d[(d.season >= 2022) & (d.season <= 2025)][feats["priors + market"]], d[(d.season >= 2022) & (d.season <= 2025)][target])
    print("  coefs:", dict(zip(feats["priors + market"], np.round(m.coef_, 3))), "intercept", round(m.intercept_, 2))
