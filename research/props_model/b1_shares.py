"""Which usage history predicts next-game target/carry share best? Time-forward, given the player plays."""
import pandas as pd, numpy as np, itertools
pg = pd.read_parquet("player_games.parquet")
pg = pg[(pg.offense_snaps.fillna(0) > 0) & pg.season.between(2021, 2025)].copy()
pg["snap"] = pg.offense_pct.fillna(0)
pg["ord"] = pg.season * 24 + pg.week
pg = pg.sort_values("ord")
POS_PRIOR_T = pg.groupby("position").tshare.mean().to_dict()
POS_PRIOR_C = pg.groupby("position").cshare.mean().to_dict()

def features(h):
    """EW share history on the current team (all prior games incl. last season), per player-game."""
    out = {}
    for (pid, team), d in pg.groupby(["pid", "posteam"]):
        o = d.ord.values; ts = d.tshare.values; cs = d.cshare.values; sn = d.snap.values
        tpsn = np.where(sn > 0, d.targets.values / np.maximum(d.t_targets.values * sn, 1e-9), 0)  # targets per (team target x snap)
        for j, idx in enumerate(d.index):
            if j == 0:
                out[idx] = (np.nan, np.nan, np.nan, np.nan, 0.0, np.nan); continue
            w = 0.5 ** ((o[j] - o[:j]) / h)
            W = w.sum()
            out[idx] = ((w * ts[:j]).sum() / W, (w * cs[:j]).sum() / W, (w * sn[:j]).sum() / W,
                        (w * sn[:j] * tpsn[:j]).sum() / max((w * sn[:j]).sum(), 1e-9), W, sn[j - 1])
    f = pd.DataFrame.from_dict(out, orient="index", columns=["h_t", "h_c", "h_snap", "h_tps", "W", "last_snap"])
    return pg.join(f)

ev = lambda d, pred, col: np.mean(np.abs(pred - d[col]))
for h in (3, 5, 8, 12, 20):
    d = features(h)
    d = d[d.season >= 2023]
    d = d[d.W > 0]
    res = []
    for k in (0.5, 1, 2, 4):
        pt = (d.h_t * d.W + k * d.position.map(POS_PRIOR_T)) / (d.W + k)
        pc = (d.h_c * d.W + k * d.position.map(POS_PRIOR_C)) / (d.W + k)
        res.append((k, ev(d, pt, "tshare"), ev(d[d.position == "RB"], pc[d.position == "RB"], "cshare")))
    print(f"half-life {h:>2} games:", "  ".join(f"k={k}: tgt {a:.4f} car {b:.4f}" for k, a, b in res))
