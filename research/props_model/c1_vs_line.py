import json, sys, statistics as st
from collections import defaultdict
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from nflmodel import prop_distributions as pd
rows = json.load(open("joined_2026.json"))
by = defaultdict(list)
for r in rows: by[r["metric"]].append(r)
print(f"{'metric':16s} {'n':>4s} {'MAE line':>8s} {'MAE proj':>8s} {'proj-line':>9s} {'act-line':>8s} {'w*':>5s} {'side hit':>8s} {'|gap|>=q hit':>12s}")
allside=[0,0]
for m, xs in sorted(by.items()):
    n = len(xs)
    ml = st.mean(abs(r["act"] - r["line"]) for r in xs)
    mp = st.mean(abs(r["act"] - r["proj"]) for r in xs)
    bias = st.mean(r["proj"] - r["line"] for r in xs)
    actb = st.mean(r["act"] - r["line"] for r in xs)
    # optimal blend weight on proj: minimize squared err of line + w*(proj-line)
    num = sum((r["proj"] - r["line"]) * (r["act"] - r["line"]) for r in xs)
    den = sum((r["proj"] - r["line"]) ** 2 for r in xs)
    w = num / den if den else 0
    dec = [r for r in xs if r["act"] != r["line"] and r["proj"] != r["line"]]
    hit = sum((r["proj"] > r["line"]) == (r["act"] > r["line"]) for r in dec) / max(len(dec),1)
    allside[0]+=sum((r["proj"] > r["line"]) == (r["act"] > r["line"]) for r in dec); allside[1]+=len(dec)
    gaps = sorted(abs(r["proj"] - r["line"]) for r in dec)
    q = gaps[int(len(gaps) * 0.75)] if gaps else 0
    big = [r for r in dec if abs(r["proj"] - r["line"]) >= q]
    hb = sum((r["proj"] > r["line"]) == (r["act"] > r["line"]) for r in big) / max(len(big),1)
    print(f"{m:16s} {n:4d} {ml:8.2f} {mp:8.2f} {bias:+9.2f} {actb:+8.2f} {w:5.2f} {hit:8.3f} {hb:8.3f} (n={len(big)})")
print("all side hit", allside, allside[0]/allside[1])
# over/under base rates
o = [r for r in rows if r["act"] != r["line"]]
print("overs hit share", sum(r["act"] > r["line"] for r in o) / len(o))
# best_bets rule simulation: P(over) from distribution vs 50% (no prices; -110 assumed)
res = defaultdict(lambda: [0, 0])
for r in rows:
    d = pd.distribution(r["metric"], r["proj"])
    if not d: continue
    p = pd.over_probability(d, r["line"])
    if p is None or r["act"] == r["line"]: continue
    for side, prob in (("over", p), ("under", 1 - p)):
        if prob >= 0.55 and prob - 0.5 >= 0.04:
            won = (r["act"] > r["line"]) == (side == "over")
            res[(r["metric"], side)][0] += won; res[(r["metric"], side)][1] += 1
tot = [0, 0]
for k, (wn, n) in sorted(res.items()):
    tot[0] += wn; tot[1] += n
    print("rule", k, f"{wn}/{n} = {wn/n:.3f}")
print("rule total", tot, tot[0] / tot[1], "units@-110", round(tot[0] * 0.909 - (tot[1] - tot[0]), 1))
