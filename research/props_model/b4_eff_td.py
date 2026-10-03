import pandas as pd, numpy as np
pg = pd.read_parquet("player_games.parquet"); pg = pg[pg.season.between(2021, 2025)]
# Efficiency reliability: split-half by season per player; pseudo-count k = n_half*(1-r)/r in units of opportunities
def rel(pos, num, den, mind=40):
    rs = []; ns = []
    for s, d in pg[pg.position == pos].groupby("season"):
        a = d[d.week % 2 == 1].groupby("pid")[[num, den]].sum(); b = d[d.week % 2 == 0].groupby("pid")[[num, den]].sum()
        j = a.join(b, lsuffix="_a", rsuffix="_b"); j = j[(j[den + "_a"] >= mind) & (j[den + "_b"] >= mind)]
        rs.append((j[num + "_a"] / j[den + "_a"]).corr(j[num + "_b"] / j[den + "_b"])); ns.append(((j[den + "_a"] + j[den + "_b"]) / 2).mean())
    r = np.nanmean(rs); n = np.mean(ns); return r, n, n * (1 - r) / r
print("efficiency: split-half r, mean opps per half, pseudo-count k (opportunities)")
for pos, num, den, prod in (("WR", "rec_yds", "targets", 38), ("TE", "rec_yds", "targets", 34), ("RB", "rec_yds", "targets", 30),
                            ("WR", "receptions", "targets", 34), ("TE", "receptions", "targets", 30), ("RB", "receptions", "targets", 28),
                            ("RB", "rush_yds", "carries", 45), ("WR", "rec_td", "targets", 55), ("RB", "rush_td", "carries", 55), ("WR", "air", "targets", None)):
    r, n, k = rel(pos, num, den)
    print(f"  {pos} {num}/{den}: r={r:.2f} n={n:.0f} k={k:.0f}  (production pseudo {prod})")
# TD model: next-game anytime TD from (a) TD rate history x opportunities vs (b) RZ/inside-10 share
pg = pg.sort_values(["season", "week"])
pg["td"] = ((pg.rec_td + pg.rush_td) > 0).astype(float)
pg["opps"] = pg.targets + pg.carries
g = pg.groupby(["pid", "posteam"])
for c in ("rz_tgt", "rz_car", "i10_tgt", "i10_car", "targets", "carries", "rec_td", "rush_td", "td"):
    pg["h_" + c] = g[c].transform(lambda s: s.shift(1).ewm(halflife=6, min_periods=1).mean())
pg["n"] = g.cumcount()
d = pg[(pg.n >= 3) & pg.position.isin(["WR", "TE", "RB"]) & (pg.season >= 2023)].dropna(subset=["h_targets"])
tr = pg[(pg.n >= 3) & pg.position.isin(["WR", "TE", "RB"]) & pg.season.between(2021, 2022)].dropna(subset=["h_targets"])
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
sets = {"td history only": ["h_td"], "volume (tgt, car)": ["h_targets", "h_carries"],
        "volume + td rate": ["h_targets", "h_carries", "h_rec_td", "h_rush_td"],
        "red zone usage": ["h_rz_tgt", "h_rz_car", "h_i10_tgt", "h_i10_car"],
        "volume + red zone": ["h_targets", "h_carries", "h_rz_tgt", "h_rz_car", "h_i10_tgt", "h_i10_car"],
        "all": ["h_targets", "h_carries", "h_rz_tgt", "h_rz_car", "h_i10_tgt", "h_i10_car", "h_rec_td", "h_rush_td"]}
print("\nanytime TD (players who played), train 2021-22 test 2023-25: Brier / logloss")
for name, f in sets.items():
    m = LogisticRegression(max_iter=2000).fit(tr[f], tr.td); p = m.predict_proba(d[f])[:, 1]
    print(f"  {name:22s} {brier_score_loss(d.td, p):.4f}  {log_loss(d.td, p):.4f}")
