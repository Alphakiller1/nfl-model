import pandas as pd, numpy as np, itertools
t = pd.read_parquet("team_games.parquet"); t = t[t.season <= 2025].sort_values(["season", "week"])
def run(metric, key, h, k_prior, k_carry):
    """pred = (sum w_i x_i + k_prior*prior)/(sum w_i + k_prior); prior = league + carry*(last season team mean - league)."""
    errs = []
    lg = t.groupby("season")[metric].mean()
    last = t.groupby(["season", key])[metric].mean()
    for (s, team), d in t.groupby(["season", key]):
        if s == 2021: continue
        prior = lg[s - 1] + k_carry * (last.get((s - 1, team), lg[s - 1]) - lg[s - 1])
        xs = d[metric].values; wk = d.week.values
        for i in range(len(xs)):
            if np.isnan(xs[i]): continue
            w = 0.5 ** ((wk[i] - wk[:i]) / h) if h else np.ones(i)
            v = xs[:i]; ok = ~np.isnan(v)
            pred = (np.sum(w[ok] * v[ok]) + k_prior * prior) / (np.sum(w[ok]) + k_prior)
            errs.append((pred - xs[i]) ** 2)
    return np.sqrt(np.mean(errs))
for metric, key in [("n_proe", "posteam"), ("n_sec", "posteam"), ("pass_att", "posteam"), ("designed_rush", "posteam"), ("plays", "posteam"),
                    ("pass_att", "defteam"), ("designed_rush", "defteam"), ("plays", "defteam")]:
    best = None
    for h, kp, kc in itertools.product([4, 8, 12, 20, 0], [2, 4, 6, 10, 20, 40], [0.0, 0.3, 0.5, 0.7]):
        e = run(metric, key, h, kp, kc)
        if best is None or e < best[0]: best = (e, h, kp, kc)
    base = run(metric, key, 0, 1e9, 0.0)  # league mean only
    print(f"{metric:14s} {key:8s} best rmse {best[0]:.3f} (half-life {best[1] or 'inf'}, prior games {best[2]}, carry {best[3]})  league-only {base:.3f}")
