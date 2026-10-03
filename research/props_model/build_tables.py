"""Team-game and player-game tables from nflverse pbp (2021-2026) for the script/pace/usage study."""
import numpy as np, pandas as pd, pyarrow.parquet as pq

COLS = ['game_id', 'season', 'week', 'season_type', 'posteam', 'defteam', 'home_team', 'away_team',
        'play_type', 'pass', 'rush', 'qb_dropback', 'qb_scramble', 'qb_kneel', 'qb_spike', 'pass_attempt',
        'rush_attempt', 'sack', 'xpass', 'pass_oe', 'score_differential', 'wp', 'vegas_wp', 'qtr',
        'half_seconds_remaining', 'game_seconds_remaining', 'down', 'ydstogo', 'yardline_100', 'spread_line',
        'total_line', 'result', 'total', 'no_huddle', 'fixed_drive', 'epa', 'success', 'receiver_player_id',
        'rusher_player_id', 'passer_player_id', 'touchdown', 'pass_touchdown', 'rush_touchdown',
        'two_point_attempt', 'play_deleted', 'aborted_play', 'air_yards', 'yards_gained',
        'complete_pass', 'roof', 'wind', 'home_coach', 'away_coach', 'div_game', 'interception', 'fumble_lost']
frames = []
for s in range(2021, 2027):
    d = pq.read_table(f"pbp_{s}.parquet", columns=COLS).to_pandas()
    frames.append(d[d.season_type == "REG"])
p = pd.concat(frames, ignore_index=True)
p = p[p.posteam.notna() & p.play_type.isin(["pass", "run"]) & (p.two_point_attempt != 1)]
p["kneel"] = p.qb_kneel.fillna(0) == 1
p["spike"] = p.qb_spike.fillna(0) == 1
p["real"] = ~p.kneel & ~p.spike
p["home"] = p.posteam == p.home_team
# team-perspective market numbers (spread_line > 0 = home favoured)
p["t_spread"] = np.where(p.home, p.spread_line, -p.spread_line)
p["neutral"] = (p.wp.between(0.2, 0.8)) & (p.qtr <= 3) & (p.down.isin([1, 2])) & p.real
# pace: seconds elapsed from this snap to the next snap of the same drive (no-huddle + clock)
p = p.sort_values(["game_id", "fixed_drive", "game_seconds_remaining"], ascending=[True, True, False])
p["next_gsr"] = p.groupby(["game_id", "fixed_drive"]).game_seconds_remaining.shift(-1)
p["sec_play"] = (p.game_seconds_remaining - p.next_gsr).where(lambda x: (x > 0) & (x < 60))
p.to_parquet("plays.parquet")

g = p.groupby(["game_id", "season", "week", "posteam", "defteam", "home"])
t = g.agg(
    plays=("real", "sum"),
    dropbacks=("qb_dropback", "sum"),
    pass_att=("pass_attempt", "sum"),
    sacks=("sack", "sum"),
    scrambles=("qb_scramble", "sum"),
    rush_att=("rush_attempt", "sum"),
    kneels=("kneel", "sum"),
    t_spread=("t_spread", "first"),
    total_line=("total_line", "first"),
    result=("result", "first"),
    game_total=("total", "first"),
    epa=("epa", "mean"),
    drives=("fixed_drive", "nunique"),
    mean_sd=("score_differential", "mean"),
    coach_h=("home_coach", "first"), coach_a=("away_coach", "first"),
    roof=("roof", "first"), wind=("wind", "first"),
).reset_index()
t["pass_att"] = t.pass_att - t.sacks  # nflverse player 'attempts' excludes sacks
nt = p[p.neutral].groupby(["game_id", "posteam"]).agg(
    n_plays=("real", "sum"), n_db=("qb_dropback", "mean"), n_proe=("pass_oe", "mean"),
    n_xpass=("xpass", "mean"), n_sec=("sec_play", "median"), n_nohuddle=("no_huddle", "mean")).reset_index()
t = t.merge(nt, on=["game_id", "posteam"], how="left")
t["team_margin"] = np.where(t.home, t.result, -t.result)
t["coach"] = np.where(t.home, t.coach_h, t.coach_a)
t["implied"] = (t.total_line + t.t_spread) / 2
t["carries"] = t.rush_att  # includes scrambles? nflverse rush_attempt=1 on scrambles; player-stat carries excludes scrambles
t["designed_rush"] = t.rush_att - t.scrambles
t.drop(columns=["coach_h", "coach_a"], inplace=True)
t.to_parquet("team_games.parquet")
print(t.shape)
print(t.groupby("season")[["plays", "dropbacks", "pass_att", "rush_att", "designed_rush", "n_db", "n_proe", "n_sec"]].mean().round(2))
