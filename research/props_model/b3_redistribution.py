import pandas as pd, numpy as np
import b1_shares as b
d = b.features(6)
pg = d.copy()
# team-game roster of who played; for each team-game, absent "regulars": played team's previous game, h_t>=.08 (or h_c>=.2), not playing now
pg["tg"] = pg.game_id + "|" + pg.posteam
games = pg[["posteam", "ord", "game_id"]].drop_duplicates().sort_values("ord")
prev_game = {}
for team, g in games.groupby("posteam"):
    ids = g.game_id.tolist()
    for a, c in zip(ids[:-1], ids[1:]): prev_game[(team, c)] = a
played = pg.groupby(["game_id", "posteam"]).pid.apply(set).to_dict()
last_h = pg.set_index(["game_id", "posteam", "pid"])[["h_t", "h_c", "position", "tshare", "cshare"]]
rows = []
for (gid, team), ps in played.items():
    pgid = prev_game.get((team, gid))
    if not pgid: continue
    prev = played.get((pgid, team), set())
    gone = prev - ps
    # each departed regular's share going into last game (h_t from last game index + last observed)
    gone_t = {}; gone_c = {}
    for pid in gone:
        try: r = last_h.loc[(pgid, team, pid)]
        except KeyError: continue
        ht = np.nanmean([r.h_t, r.tshare]) if not np.isnan(r.h_t) else r.tshare
        hc = np.nanmean([r.h_c, r.cshare]) if not np.isnan(r.h_c) else r.cshare
        gone_t[pid] = (ht, r.position); gone_c[pid] = (hc, r.position)
    rows.append((gid, team, sum(v for v, _ in gone_t.values()), sum(v for v, _ in gone_c.values()),
                 {p: sum(v for v, pp in gone_t.values() if pp == p) for p in ("WR", "TE", "RB")}))
ab = pd.DataFrame(rows, columns=["game_id", "posteam", "gone_t", "gone_c", "gone_t_pos"])
x = pg.merge(ab, on=["game_id", "posteam"], how="left").dropna(subset=["h_t", "gone_t"])
x = x[(x.season >= 2022)]
x["gone_same"] = [r.get(p, 0) for r, p in zip(x.gone_t_pos, x.position)]
big = x[x.gone_t >= 0.12]
print("team-games with a >=12% target-share regular missing:", big.groupby(["game_id", "posteam"]).ngroups)
for name, pred in (("no redistribution", big.h_t),
                   ("proportional", big.h_t / (1 - big.gone_t).clip(lower=0.4)),
                   ("half proportional", big.h_t / (1 - 0.5 * big.gone_t).clip(lower=0.4))):
    print(f"  target share MAE {name:20s} {np.mean(np.abs(pred - big.tshare)):.4f}   bias {np.mean(pred - big.tshare):+.4f}")
# fit: tshare = h_t*(1 + a*gone_t) + b*gone_same * h_t/(sum h_t same pos)?
from sklearn.linear_model import LinearRegression
big = big.assign(i1=big.h_t * big.gone_t, i2=big.h_t * big.gone_same)
m = LinearRegression(fit_intercept=False).fit(big[["h_t", "i1", "i2"]], big.tshare)
print("  fitted: share = h_t*(%.3f + %.3f*gone_all + %.3f*gone_same_position)" % tuple(m.coef_))
bc = x[(x.gone_c >= 0.25) & (x.position == "RB")]
print("RB team-games with a >=25% carry-share back missing:", len(bc))
for name, pred in (("no redistribution", bc.h_c), ("proportional", bc.h_c / (1 - bc.gone_c).clip(lower=0.3))):
    print(f"  carry share MAE {name:20s} {np.mean(np.abs(pred - bc.cshare)):.4f}   bias {np.mean(pred - bc.cshare):+.4f}")
