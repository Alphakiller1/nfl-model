"""Point-in-time feature table for the player prop matrix (player-games, 2022-2026).

Every feature uses only games before the one being predicted. Coverage profiles
(man/zone) come from the PREVIOUS season's participation file, which is what
is available at prediction time (2026 participation is not published).
"""
import numpy as np, pandas as pd, pyarrow.parquet as pq

pg = pd.read_parquet("player_games.parquet")
tg = pd.read_parquet("team_games_feat.parquet")
plays = pd.read_parquet("plays.parquet")
plays = plays[plays.real]

# ---------- opponent for each player-game
opp = tg[["game_id", "posteam", "defteam", "t_spread", "total_line", "home", "o_pass", "d_pass",
          "o_rush", "d_rush", "season", "week"]].rename(columns={"season": "s2", "week": "w2"})
pg = pg.merge(opp, on=["game_id", "posteam"], how="inner")
pg = pg[(pg.offense_snaps.fillna(0) > 0) & pg.season.between(2021, 2026)].copy()
pg["ord"] = pg.season * 24 + pg.week

# ---------- pass-play participation: routes on the field, man/zone splits (2022-2025)
parts = []
for s in range(2022, 2026):
    d = pq.read_table(f"part_{s}.parquet", columns=["nflverse_game_id", "play_id", "possession_team",
                                                   "offense_players", "defense_man_zone_type", "was_pressure"]).to_pandas()
    d["season"] = s
    parts.append(d)
part = pd.concat(parts)
part = part.rename(columns={"nflverse_game_id": "game_id"})
pp = plays[["game_id", "play_id", "posteam", "defteam", "qb_dropback", "receiver_player_id", "yards_gained",
            "complete_pass", "air_yards", "season", "week"]].copy()
pp["play_id"] = pp.play_id.astype(float)
part["play_id"] = part.play_id.astype(float)
dp = pp[pp.qb_dropback == 1].merge(part[["game_id", "play_id", "offense_players", "defense_man_zone_type"]],
                                   on=["game_id", "play_id"], how="inner")
dp["cvg"] = dp.defense_man_zone_type.map({"MAN_COVERAGE": "man", "ZONE_COVERAGE": "zone"})
ex = dp[["game_id", "season", "week", "posteam", "defteam", "cvg", "receiver_player_id", "yards_gained",
         "complete_pass", "offense_players"]].copy()
ex["pid"] = ex.offense_players.str.split(";")
ex = ex.explode("pid")
ex = ex[ex.pid.notna() & (ex.pid != "")]
ex["tgt"] = (ex.receiver_player_id == ex.pid).astype(float)
ex["yds"] = np.where((ex.tgt == 1) & (ex.complete_pass == 1), ex.yards_gained, 0.0)
routes = ex.groupby(["game_id", "pid"]).agg(pass_snaps=("tgt", "size")).reset_index()
pg = pg.merge(routes, on=["game_id", "pid"], how="left")
# prior-season coverage profile per player: targets per pass snap vs man and vs zone
cv = ex[ex["cvg"].notna()].groupby(["season", "pid", "cvg"]).agg(n=("tgt", "size"), t=("tgt", "sum"),
                                                                  y=("yds", "sum")).reset_index()
cvw = cv.pivot_table(index=["season", "pid"], columns="cvg", values=["n", "t", "y"], fill_value=0)
cvw.columns = [f"{a}_{b}" for a, b in cvw.columns]
cvw = cvw.reset_index()
K_RT = 60.0  # pass snaps of shrinkage toward the player's own pooled rate
pool = (cvw.t_man + cvw.t_zone) / (cvw.n_man + cvw.n_zone).clip(lower=1)
cvw["tprr_man"] = (cvw.t_man + K_RT * pool) / (cvw.n_man + K_RT)
cvw["tprr_zone"] = (cvw.t_zone + K_RT * pool) / (cvw.n_zone + K_RT)
cvw["man_tilt"] = np.log(cvw.tprr_man.clip(1e-3) / cvw.tprr_zone.clip(1e-3))  # >0: earns more vs man
cvw["season"] = cvw.season + 1  # used the following season
pg = pg.merge(cvw[["season", "pid", "man_tilt"]], on=["season", "pid"], how="left")
# prior-season defensive man rate
dm = dp[dp["cvg"].notna()].groupby(["season", "defteam"]).cvg.apply(lambda s: (s == "man").mean()).reset_index(name="opp_man")
lg_man = dm.groupby("season").opp_man.transform("mean")
dm["opp_man_dev"] = dm.opp_man - lg_man
dm["season"] = dm.season + 1
pg = pg.merge(dm[["season", "defteam", "opp_man_dev"]], on=["season", "defteam"], how="left")

# ---------- FTN: defensive blitz rate and box counts faced, point-in-time
ftn = pd.concat([pq.read_table(f"ftn_{s}.parquet", columns=["nflverse_game_id", "nflverse_play_id", "season", "week",
                                                           "n_blitzers", "n_defense_box"]).to_pandas() for s in range(2022, 2027)])
ftn = ftn.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
ftn["play_id"] = ftn.play_id.astype(float)
fp = pp.merge(ftn[["game_id", "play_id", "n_blitzers", "n_defense_box"]], on=["game_id", "play_id"], how="inner")
fp["blitz"] = (fp.n_blitzers.fillna(0) > 0).astype(float)
fp["heavy_box"] = (fp.n_defense_box.fillna(0) >= 8).astype(float)
ftn_def = fp.groupby(["game_id", "defteam"]).agg(blitz=("blitz", lambda s: s[fp.loc[s.index, "qb_dropback"] == 1].mean()),
                                                  heavy_box=("heavy_box", lambda s: s[fp.loc[s.index, "qb_dropback"] == 0].mean())).reset_index()

# ---------- per-game defensive allowed profile (by position group) and pressure
pl_pos = pg[["pid", "position"]].drop_duplicates("pid")
pt = plays[(plays.pass_attempt == 1) & (plays.sack != 1)].merge(pl_pos.rename(columns={"pid": "receiver_player_id"}),
                                                                  on="receiver_player_id", how="left")
pt["grp"] = pt.position.replace({"FB": "RB"})
pt["cy"] = np.where(pt.complete_pass == 1, pt.yards_gained, 0.0)
dg = pt[pt.grp.isin(["WR", "TE", "RB"])].groupby(["game_id", "defteam", "grp"]).agg(
    t=("pass_attempt", "size"), y=("cy", "sum"), c=("complete_pass", "sum")).reset_index()
dtot = dg.groupby(["game_id", "defteam"]).t.transform("sum")
dg["share"] = dg.t / dtot
dgw = dg.pivot_table(index=["game_id", "defteam"], columns="grp", values=["t", "y", "c", "share"], fill_value=0)
dgw.columns = [f"{a}_{b}" for a, b in dgw.columns]
dgw = dgw.reset_index()
rr = plays[(plays.rush_attempt == 1) & (plays.qb_scramble != 1)]
dr = rr.groupby(["game_id", "defteam"]).agg(rc=("rush_attempt", "size"), ry=("yards_gained", "sum"),
                                            repa=("epa", "sum")).reset_index()
db = plays[plays.qb_dropback == 1].copy()
db["press"] = ((db.sack == 1) | (db.qb_hit == 1)).astype(float)  # pressure proxy available every season
dpb = db.groupby(["game_id", "defteam"]).agg(db=("qb_dropback", "size"), sk=("press", "sum"), pepa=("epa", "sum")).reset_index()
opb = db.groupby(["game_id", "posteam"]).agg(odb=("qb_dropback", "size"), osk=("press", "sum"), opepa=("epa", "sum")).reset_index()
orr = rr.groupby(["game_id", "posteam"]).agg(orc=("rush_attempt", "size"), ory=("yards_gained", "sum"), orepa=("epa", "sum")).reset_index()
gdate = tg[["game_id", "season", "week"]].drop_duplicates()
D = dgw.merge(dr, on=["game_id", "defteam"], how="outer").merge(dpb, on=["game_id", "defteam"], how="outer") \
       .merge(ftn_def, on=["game_id", "defteam"], how="left").merge(gdate, on="game_id")
O = opb.merge(orr, on=["game_id", "posteam"], how="outer").merge(gdate, on="game_id")

def pit_team(df, key, sums, h=4.0, k=10.0, carry=0.3):
    """EW (half-life h games) ratio-of-sums per team, shrunk with k pseudo-games toward a prior
    of league + carry*(last season - league). Returns per (game_id, team) the pre-game value."""
    df = df.sort_values(["season", "week"])
    out = []
    league = {}
    for s, d in df.groupby("season"):
        league[s] = {n: d[a].sum() / max(d[b].sum(), 1e-9) for n, (a, b) in sums.items()}
    lastseason = {}
    for (s, team), d in df.groupby(["season", key]):
        lastseason[(s, team)] = {n: d[a].sum() / max(d[b].sum(), 1e-9) for n, (a, b) in sums.items()}
    for (s, team), d in df.groupby(["season", key]):
        base = league.get(s - 1, league[s])
        last = lastseason.get((s - 1, team))
        wk = d.week.values
        for j, gid in enumerate(d.game_id.values):
            w = 0.5 ** ((wk[j] - wk[:j]) / h)
            row = {"game_id": gid, key: team}
            for n, (a, b) in sums.items():
                prior = base[n] + (carry * (last[n] - base[n]) if last else 0.0)
                num = (w * d[a].values[:j]).sum(); den = (w * d[b].values[:j]).sum()
                gw = w.sum()
                # k pseudo-games of an average game's denominator
                avg_den = d[b].values[:j].mean() if j else df[b].mean()
                row[n] = (num + k * avg_den * prior) / (den + k * avg_den)
                row[n + "_lg"] = base[n]
            out.append(row)
    return pd.DataFrame(out)

Dp = pit_team(D.fillna(0), "defteam", {
    "d_share_WR": ("t_WR", "db"), "d_share_TE": ("t_TE", "db"), "d_share_RB": ("t_RB", "db"),
    "d_ypt_WR": ("y_WR", "t_WR"), "d_ypt_TE": ("y_TE", "t_TE"), "d_ypt_RB": ("y_RB", "t_RB"),
    "d_cr_WR": ("c_WR", "t_WR"), "d_cr_TE": ("c_TE", "t_TE"), "d_cr_RB": ("c_RB", "t_RB"),
    "d_ypc": ("ry", "rc"), "d_repa": ("repa", "rc"), "d_pepa": ("pepa", "db"), "d_sack": ("sk", "db")})
# blitz/box are rates already: weight by games
D["one"] = 1.0
D["blitz_x"] = D.blitz.fillna(D.blitz.mean()); D["box_x"] = D.heavy_box.fillna(D.heavy_box.mean())
Db = pit_team(D, "defteam", {"d_blitz": ("blitz_x", "one"), "d_box": ("box_x", "one")})
Op = pit_team(O.fillna(0), "posteam", {"o_sack": ("osk", "odb"), "o_pepa": ("opepa", "odb"),
                                         "o_ypc": ("ory", "orc"), "o_repa": ("orepa", "orc")}, k=6.0, carry=0.4)
pg = pg.merge(Dp, on=["game_id", "defteam"], how="left").merge(Db, on=["game_id", "defteam"], how="left") \
       .merge(Op, on=["game_id", "posteam"], how="left")

# ---------- player priors (strictly before the game)
pg = pg.sort_values(["pid", "ord"]).reset_index(drop=True)
def ew(df, by, col, h, weight=None):
    """Exponentially weighted mean over previous rows within group `by` (half-life in games)."""
    a = 0.5 ** (1.0 / h)
    out = np.full(len(df), np.nan)
    for _, idx in df.groupby(by).indices.items():
        num = den = 0.0
        x = df[col].values[idx]
        w8 = np.ones(len(idx)) if weight is None else df[weight].values[idx]
        for j, i in enumerate(idx):
            out[i] = num / den if den > 0 else np.nan
            if not np.isnan(x[j]):
                num = a * num + w8[j] * x[j]; den = a * den + w8[j]
            else:
                num *= a; den *= a
    return out

pg["snap"] = pg.offense_pct.fillna(0)
pg["tpr"] = pg.targets / pg.pass_snaps
same = ["pid", "posteam"]
pg["p_tshare"] = ew(pg, same, "tshare", 6)
pg["p_cshare"] = ew(pg, same, "cshare", 6)
pg["p_snap"] = ew(pg, same, "snap", 6)
pg["p_last_snap"] = pg.groupby(same).snap.shift(1)
pg["p_games"] = pg.groupby(same).cumcount()
# efficiency, all teams, long memory (weight by opportunities)
for c in ("targets", "air", "rec_yds", "receptions", "carries", "rush_yds", "rec_td", "rush_td"):
    pg["s_" + c] = ew(pg, "pid", c, 20)
pg.to_parquet("matrix_features.parquet")
print(pg.shape)
print(pg[["p_tshare", "p_cshare", "man_tilt", "opp_man_dev", "d_ypt_WR", "d_blitz", "d_box", "o_sack", "pass_snaps"]].describe().T.round(3))
