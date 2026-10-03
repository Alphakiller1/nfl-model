import pandas as pd, numpy as np
p = pd.read_parquet("plays.parquet"); p = p[p.real & (p.pass_attempt == 1) & (p.sack != 1) & p.passer_player_id.notna() & p.season.between(2021, 2025)]
p["cy"] = np.where(p.complete_pass == 1, p.yards_gained, 0)
q = p.groupby(["season", "week", "passer_player_id"]).agg(att=("pass_attempt", "size"), comp=("complete_pass", "sum"), yds=("cy", "sum"),
    td=("pass_touchdown", "sum"), ints=("interception", "sum"), air=("air_yards", "sum")).reset_index()
for num, prod in (("comp", 75), ("yds", 75), ("td", 90), ("ints", 100), ("air", None)):
    rs, ns = [], []
    for s, d in q.groupby("season"):
        a = d[d.week % 2 == 1].groupby("passer_player_id")[[num, "att"]].sum(); b = d[d.week % 2 == 0].groupby("passer_player_id")[[num, "att"]].sum()
        j = a.join(b, lsuffix="_a", rsuffix="_b"); j = j[(j.att_a >= 120) & (j.att_b >= 120)]
        rs.append((j[num + "_a"] / j.att_a).corr(j[num + "_b"] / j.att_b)); ns.append(((j.att_a + j.att_b) / 2).mean())
    r, n = np.mean(rs), np.mean(ns)
    print(f"QB {num}/att: r={r:.2f} n={n:.0f} k={n*(1-r)/r:.0f} attempts (production {prod})")
