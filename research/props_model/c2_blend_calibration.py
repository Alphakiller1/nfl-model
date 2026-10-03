import json, sys, math, statistics as st, random
from collections import defaultdict
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from nflmodel import prop_distributions as pd
rows = json.load(open("joined_2026.json"))
def wfit(xs):
    num = sum((r["proj"] - r["line"]) * (r["act"] - r["line"]) for r in xs)
    den = sum((r["proj"] - r["line"]) ** 2 for r in xs)
    return num / den if den else 0
by = defaultdict(list)
for r in rows: by[r["metric"]].append(r)
random.seed(1)
print("blend weight with game-clustered bootstrap 90% CI")
for m, xs in sorted(by.items()):
    games = defaultdict(list)
    for r in xs: games[r["event"]].append(r)
    keys = list(games)
    bs = []
    for _ in range(500):
        s = [r for k in random.choices(keys, k=len(keys)) for r in games[k]]
        bs.append(wfit(s))
    bs.sort()
    print(f"  {m:16s} w={wfit(xs):.2f}  [{bs[25]:.2f}, {bs[475]:.2f}]")
# Brier / calibration of P(over)
def brier(ps): return st.mean((p - y) ** 2 for p, y in ps)
P = defaultdict(list)
for r in rows:
    if r["act"] == r["line"]: continue
    d = pd.distribution(r["metric"], r["proj"])
    if not d: continue
    p = pd.over_probability(d, r["line"]); y = float(r["act"] > r["line"])
    P[r["metric"]].append((p, y))
print("\nBrier: model P(over) vs coin 0.5")
for m, ps in sorted(P.items()):
    mp = st.mean(p for p, y in ps); my = st.mean(y for p, y in ps)
    print(f"  {m:16s} n={len(ps)} model={brier(ps):.4f} coin={brier([(0.5,y) for p,y in ps]):.4f} mean P(over)={mp:.3f} actual over={my:.3f}")
allp = [x for v in P.values() for x in v]
print("bins:")
for lo in (0, .3, .4, .45, .5, .55, .6, .7):
    b = [(p, y) for p, y in allp if lo <= p < lo + (0.1 if lo in (0, .3, .6) else .05 if lo < .6 else 0.3)]
    if b: print(f"  P in [{lo:.2f}..) n={len(b)} pred={st.mean(p for p,y in b):.3f} act={st.mean(y for p,y in b):.3f}")
