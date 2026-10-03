import pandas as pd, numpy as np
t = pd.read_parquet("team_games.parquet")
t = t[t.season <= 2025]
t["proe_x"] = t.n_proe
t["db_rate"] = t.dropbacks / t.plays
metrics = {"n_proe": "neutral PROE", "n_db": "neutral dropback rate", "n_sec": "neutral sec/play",
           "plays": "offensive plays", "pass_att": "pass attempts", "designed_rush": "designed runs",
           "db_rate": "dropback rate (all)", "epa": "EPA/play", "drives": "drives"}
print(f"{'metric':24s} {'split-half r':>12s} {'games for 50%':>14s} {'YoY r':>6s} {'def split r':>11s} {'def k':>6s}")
for m, label in metrics.items():
    rs = []; rd = []
    for s, d in t.groupby("season"):
        for side, key in (("off", "posteam"), ("def", "defteam")):
            odd = d[d.week % 2 == 1].groupby(key)[m].mean(); even = d[d.week % 2 == 0].groupby(key)[m].mean()
            r = odd.corr(even)
            (rs if side == "off" else rd).append(r)
    r = np.mean(rs); n_half = t.groupby(["season", "posteam"]).size().mean() / 2
    # r_half = n*s/(n*s+e) -> per-game reliability ratio; k = games at which weight = .5 = e/s
    k = n_half * (1 - r) / r if r > 0 else float("inf")
    rdm = np.mean(rd); kd = n_half * (1 - rdm) / rdm if rdm > 0 else float("inf")
    tm = t.groupby(["season", "posteam"])[m].mean().unstack(0)
    yoy = np.mean([tm[s].corr(tm[s + 1]) for s in range(2021, 2025)])
    print(f"{label:24s} {r:12.3f} {k:14.1f} {yoy:6.2f} {rdm:11.3f} {kd:6.1f}")
