import json, csv, math, statistics as st
from collections import defaultdict, Counter
from dist_fit import p_over
D = json.load(open("dist_2025.json")); lad = {k: [(hi, tuple(q)) for hi, q in v] for k, v in D["ladders"].items()}; nb = D["nb"]
props = json.load(open("espn_props_2026.json"))
emap = {}
for f in ("roster_weekly_2026.csv", "roster_weekly_2025.csv"):
    for r in csv.DictReader(open(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "cache" / f), encoding="utf-8-sig")):
        if r.get("espn_id") and r.get("gsis_id"): emap.setdefault(r["espn_id"].split(".")[0], r["gsis_id"])
proj = {(r["week"], r["pid"]): r for r in json.load(open("rm_2026.json"))}
rows = []
for p in props:
    r = proj.get((p["week"], emap.get(p["espn_id"])))
    if not r or not r["played"]: continue
    m = p["metric"]
    if m == "carries" and r["pos"] == "QB": m = "rush_attempts"
    if m not in r["m"] or (m not in lad and m not in nb): continue
    a = r["a"][m]
    if a == p["line"]: continue
    rows.append({"week": p["week"], "m": m, "mu": r["m"][m], "line": p["line"], "y": float(a > p["line"]), "pos": r["pos"]})
print("lines joined", len(rows), Counter(r["week"] for r in rows))
for r in rows: r["p"] = min(max(p_over(r["m"], r["mu"], r["line"], lad, nb), 0.01), 0.99)
fam = lambda m: {"receiving_yards": "rec", "receptions": "rec", "targets": "rec", "rushing_yards": "rush", "carries": "rush", "rush_attempts": "rush"}.get(m, "pass")
by = defaultdict(list)
for r in rows: by[fam(r["m"])].append(r)
for f, xs in sorted(by.items()):
    br = st.mean((r["p"] - r["y"]) ** 2 for r in xs)
    plays = [r for r in xs if max(r["p"], 1 - r["p"]) >= 0.55]
    w = sum((r["p"] >= .5) == (r["y"] == 1) for r in plays)
    ov = sum(r["p"] >= .5 for r in plays)
    side = sum((r["p"] >= .5) == (r["y"] == 1) for r in xs)
    print(f"{f}: n={len(xs)} brier {br:.4f} (coin .25) mean p {st.mean(r['p'] for r in xs):.3f} actual over {st.mean(r['y'] for r in xs):.3f} | side hit {side/len(xs):.3f} | plays>=55% {w}-{len(plays)-w} ({w/max(len(plays),1):.3f}) overs {ov} unders {len(plays)-ov}")
allp = [r for r in rows if max(r["p"], 1 - r["p"]) >= 0.55]; w = sum((r["p"] >= .5) == (r["y"] == 1) for r in allp)
print("ALL plays>=55%:", f"{w}-{len(allp)-w} ({w/len(allp):.3f})", "brier", round(st.mean((r["p"]-r["y"])**2 for r in rows), 4))
for lo in (0.0, 0.35, 0.45, 0.55, 0.65):
    b = [r for r in rows if lo <= r["p"] < lo + (0.35 if lo == 0 else 0.1 if lo < .65 else 1)]
    if b: print(f"  p in [{lo:.2f}..) n={len(b)} pred {st.mean(r['p'] for r in b):.3f} act {st.mean(r['y'] for r in b):.3f}")
json.dump(rows, open("line_rows_2026.json", "w"))
