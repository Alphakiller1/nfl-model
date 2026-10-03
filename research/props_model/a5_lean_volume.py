import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
t = pd.read_parquet("team_games_feat.parquet")
d = t[(t.week >= 2)].dropna(subset=["o_pass", "d_pass", "o_rush", "d_rush", "t_spread", "total_line", "o_proe", "o_sec", "opp_sec"])
for target, sets in (("pass_att", [["o_pass", "d_pass", "t_spread", "total_line"], ["o_pass", "d_pass", "o_proe", "o_sec", "opp_sec", "t_spread", "total_line"], ["prod_pass"]]),
                     ("designed_rush", [["o_rush", "d_rush", "t_spread", "total_line"], ["o_rush", "d_rush", "o_proe", "o_sec", "opp_sec", "t_spread", "total_line"], ["prod_rush"]])):
    for f in sets:
        res = []
        for test in (2024, 2025):
            tr = d[(d.season >= 2022) & (d.season < test)]; te = d[d.season == test]
            if f[0].startswith("prod"): pr = te[f[0]]
            else: pr = LinearRegression().fit(tr[f], tr[target]).predict(te[f])
            res.append(np.mean(np.abs(pr - te[target]))); 
        b = d[(d.season >= 2022) & (d.season <= 2025)]
        coef = "" if f[0].startswith("prod") else dict(zip(f, np.round(LinearRegression().fit(b[f], b[target]).coef_, 3))) 
        ic = "" if f[0].startswith("prod") else round(LinearRegression().fit(b[f], b[target]).intercept_, 2)
        print(target, "+".join(f), "MAE 2024 %.3f 2025 %.3f" % tuple(res), coef, ic)
print("league means 2022-25: pass_att %.2f designed %.2f total_line %.2f" % (d.pass_att.mean(), d.designed_rush.mean(), d.total_line.mean()))
