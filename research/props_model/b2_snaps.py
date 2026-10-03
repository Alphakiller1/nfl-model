import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
import b1_shares as b
d = b.features(6)
d = d[(d.W > 0)]
d["lag_t"] = d.groupby(["pid", "posteam"]).tshare.shift(1)
d["lag_c"] = d.groupby(["pid", "posteam"]).cshare.shift(1)
d = d.dropna(subset=["h_t", "h_snap", "last_snap", "lag_t", "h_tps"])
tr = d[d.season.between(2021, 2024)]; te = d[d.season == 2025]
for target, sets in (("tshare", [["h_t"], ["h_t", "lag_t"], ["h_t", "h_snap"], ["h_t", "h_snap", "last_snap"], ["h_t", "lag_t", "h_snap", "last_snap", "h_tps"]]),
                     ("cshare", [["h_c"], ["h_c", "lag_c"], ["h_c", "h_snap", "last_snap"], ["h_c", "lag_c", "h_snap", "last_snap"]])):
    for pos in (["WR", "TE", "RB"] if target == "tshare" else ["RB"]):
        a = tr[tr.position == pos]; e = te[te.position == pos]
        line = f"{target} {pos}:"
        for f in sets:
            m = LinearRegression().fit(a[f], a[target]); line += f"  {'+'.join(f)} {np.mean(np.abs(m.predict(e[f]) - e[target])):.4f}"
        print(line)
