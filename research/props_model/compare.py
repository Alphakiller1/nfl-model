import json, sys, statistics as st, math
from collections import defaultdict
A = {(r["week"], r["pid"]): r for r in json.load(open(sys.argv[1]))}
B = {(r["week"], r["pid"]): r for r in json.load(open(sys.argv[2]))}
KEYS = [k for k in A if k in B]
print(f"common rows {len(KEYS)}  (A {len(A)}, B {len(B)})")
M = {"QB": ["pass_attempts", "completions", "passing_yards", "passing_tds", "interceptions", "rush_attempts", "rushing_yards"],
     "RB": ["carries", "rushing_yards", "targets", "receptions", "receiving_yards"],
     "WR": ["targets", "receptions", "receiving_yards"], "TE": ["targets", "receptions", "receiving_yards"], "K": ["fg_made", "kicking_points"]}
tot = defaultdict(lambda: [0, 0])
for pos, ms in M.items():
    for m in ms:
        xs = [(A[k], B[k]) for k in KEYS if A[k]["pos"] == pos and m in A[k]["m"] and m in B[k]["m"]]
        if not xs: continue
        f = lambda r: r["m"][m] - r["a"][m]
        ra = math.sqrt(st.mean(f(a) ** 2 for a, b in xs)); rb = math.sqrt(st.mean(f(b) ** 2 for a, b in xs))
        ma = st.mean(abs(f(a)) for a, b in xs); mb = st.mean(abs(f(b)) for a, b in xs)
        ba = st.mean(f(a) for a, b in xs); bb = st.mean(f(b) for a, b in xs)
        print(f"{pos} {m:16s} RMSE {ra:7.2f} -> {rb:7.2f} ({100*(rb-ra)/ra:+5.1f}%)  MAE {ma:6.2f} -> {mb:6.2f} ({100*(mb-ma)/ma:+5.1f}%)  bias {ba:+6.2f} -> {bb:+6.2f}")
def brier(D):
    xs = [D[k] for k in KEYS if "anytime_td_probability" in D[k]["m"]]
    return st.mean((r["m"]["anytime_td_probability"] - r["a"]["anytime_td_probability"]) ** 2 for r in xs), len(xs)
print("anytime TD Brier", brier(A), "->", brier(B))
