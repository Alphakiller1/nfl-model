import numpy as np, pandas as pd
import b1_shares as b
d = b.features(6).dropna(subset=["h_t", "h_c", "h_snap", "last_snap"])
d = d[(d.h_snap > 0.05) & (d.W > 0.5)]
for pos, col, h in (("WR", "tshare", "h_t"), ("TE", "tshare", "h_t"), ("RB", "tshare", "h_t"), ("RB", "cshare", "h_c")):
    x = d[d.position == pos]; tr = x[x.season <= 2024]; te = x[x.season == 2025]
    best = None
    for beta in np.arange(0, 1.01, 0.1):
        pred = lambda z: z[h] * (z.last_snap.clip(0.02) / z.h_snap) ** beta
        e = np.mean(np.abs(pred(tr) - tr[col]))
        if best is None or e < best[0]: best = (e, beta)
    beta = best[1]; pred = lambda z, bb: z[h] * (z.last_snap.clip(0.02) / z.h_snap) ** bb
    print(pos, col, "beta", round(beta, 1), "2025 MAE", round(np.mean(np.abs(pred(te, 0) - te[col])), 4), "->", round(np.mean(np.abs(pred(te, beta) - te[col])), 4))
