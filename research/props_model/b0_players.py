"""Player-game usage table from pbp + snap counts."""
import pandas as pd, numpy as np, pyarrow.parquet as pq
p = pd.read_parquet("plays.parquet")
p = p[p.real]
p["rz"] = p.yardline_100 <= 20
p["i10"] = p.yardline_100 <= 10
p["pass_play"] = p.pass_attempt == 1
p["designed"] = (p.rush_attempt == 1) & (p.qb_scramble != 1)
tg = p[p.pass_play & p.receiver_player_id.notna() & (p.sack != 1)]
rec = tg.groupby(["game_id", "season", "week", "posteam", "receiver_player_id"]).agg(
    targets=("pass_play", "size"), receptions=("complete_pass", "sum"),
    rec_yds=("yards_gained", lambda s: s[tg.loc[s.index, "complete_pass"] == 1].sum()),
    air=("air_yards", "sum"), rz_tgt=("rz", "sum"), i10_tgt=("i10", "sum"),
    rec_td=("pass_touchdown", "sum")).reset_index().rename(columns={"receiver_player_id": "pid"})
ru = p[p.designed & p.rusher_player_id.notna()]
rus = ru.groupby(["game_id", "season", "week", "posteam", "rusher_player_id"]).agg(
    carries=("designed", "size"), rush_yds=("yards_gained", "sum"), rz_car=("rz", "sum"),
    i10_car=("i10", "sum"), rush_td=("rush_touchdown", "sum")).reset_index().rename(columns={"rusher_player_id": "pid"})
pg = rec.merge(rus, on=["game_id", "season", "week", "posteam", "pid"], how="outer").fillna(0)
team = p.groupby(["game_id", "posteam"]).agg(
    t_targets=("receiver_player_id", lambda s: s.notna().sum()), t_designed=("designed", "sum"),
    t_rz_tgt=("rz", lambda s: (s & p.loc[s.index, "receiver_player_id"].notna()).sum()),
    t_rz_car=("rz", lambda s: (s & p.loc[s.index, "designed"]).sum()),
    t_plays=("real", "sum")).reset_index()
pg = pg.merge(team, on=["game_id", "posteam"], how="left")
# snaps: pfr -> gsis
pl = pq.read_table("players.parquet", columns=["gsis_id", "pfr_id", "position", "display_name"]).to_pandas()
sn = pd.concat([pq.read_table(f"snaps_{s}.parquet").to_pandas() for s in range(2021, 2027)])
sn = sn[sn.game_type == "REG"].merge(pl[["gsis_id", "pfr_id"]].dropna(), left_on="pfr_player_id", right_on="pfr_id", how="left")
sn = sn[sn.offense_snaps > 0][["game_id", "season", "week", "team", "gsis_id", "position", "offense_snaps", "offense_pct"]]
sn = sn.rename(columns={"team": "posteam", "gsis_id": "pid", "position": "snap_pos"})
sn["posteam"] = sn.posteam.replace({"LA": "LA", "OAK": "LV", "SD": "LAC", "STL": "LA"})
sn = sn.dropna(subset=["pid"])
pg = pg.merge(sn, on=["game_id", "season", "week", "posteam", "pid"], how="outer")
# players with snaps but no touches: fill zeros and team totals
tcols = ["t_targets", "t_designed", "t_rz_tgt", "t_rz_car", "t_plays"]
pg = pg.drop(columns=tcols).merge(team, on=["game_id", "posteam"], how="left")
for c in ["targets", "receptions", "rec_yds", "air", "rz_tgt", "i10_tgt", "rec_td", "carries", "rush_yds", "rz_car", "i10_car", "rush_td"]:
    pg[c] = pg[c].fillna(0)
pg = pg.merge(pl[["gsis_id", "position", "display_name"]].rename(columns={"gsis_id": "pid"}), on="pid", how="left")
pg["position"] = pg.position.fillna(pg.snap_pos)
pg = pg[pg.position.isin(["QB", "RB", "FB", "WR", "TE"])]
pg["tshare"] = pg.targets / pg.t_targets
pg["cshare"] = pg.carries / pg.t_designed
pg.to_parquet("player_games.parquet")
print(pg.shape, pg.offense_snaps.notna().mean())
print(pg.groupby("position")[["targets", "carries", "offense_pct", "tshare", "cshare"]].mean().round(3))
