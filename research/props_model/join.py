import json, csv, math, statistics as st, sys
from collections import defaultdict
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from nflmodel import prop_distributions as pd
props = json.load(open("espn_props_2026.json"))
emap = {}
for f in ("roster_weekly_2026.csv", "roster_weekly_2025.csv"):
    for r in csv.DictReader(open(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "cache") + "/" + f, encoding="utf-8-sig")):
        if r.get("espn_id") and r.get("gsis_id"):
            emap.setdefault(r["espn_id"].split(".")[0], r["gsis_id"])
led = json.load(open("ledger.json"))
proj = {(r["week"], r["player_id"]): r for r in led["player_snapshots"] if r["status"] == "graded"}
rows = []
miss = 0
for p in props:
    g = emap.get(p["espn_id"])
    r = proj.get((p["week"], g))
    if r is None or p["metric"] not in r["metrics"]:
        miss += 1; continue
    rows.append({**p, "proj": r["metrics"][p["metric"]], "act": r["actual_metrics"][p["metric"]],
                 "pos": r["position"], "rank": r["depth_rank"], "ver": r["model_version"], "player": r["player_name"]})
print("joined", len(rows), "missed", miss)
json.dump(rows, open("joined_2026.json", "w"))
