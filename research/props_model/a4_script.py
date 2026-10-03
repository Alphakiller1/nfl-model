import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
t = pd.read_parquet("team_games_feat.parquet"); t = t[(t.season.between(2022, 2025))]
t["db_rate"] = t.dropbacks / t.plays
def fit(y, X):
    m = LinearRegression().fit(t[X], t[y]); return dict(zip(X, np.round(m.coef_, 3)))
print("market-only slopes per point of team spread / total:")
for y in ["plays", "db_rate", "dropbacks", "pass_att", "designed_rush", "scrambles", "sacks", "mean_sd", "drives", "n_proe"]:
    print(f"  {y:14s}", fit(y, ["t_spread", "total_line"]))
# spread bins: how volume moves
t["sb"] = pd.cut(t.t_spread, [-30, -9.5, -6.5, -3.5, -0.5, 0.5, 3.5, 6.5, 9.5, 30])
print(t.groupby("sb", observed=True)[["plays", "db_rate", "pass_att", "designed_rush", "mean_sd"]].mean().round(2))
# realised script: how much of the pass-att variance is explained by the realised margin?
print("\nrealised margin slope (what game script does AFTER kickoff):", fit("pass_att", ["team_margin", "total_line"]), fit("designed_rush", ["team_margin", "total_line"]))
print("corr(spread, margin)", round(t.t_spread.corr(t.team_margin), 3), " sd margin|spread", round((t.team_margin - t.t_spread).std(), 2))
