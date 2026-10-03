import json, csv, gzip, io, statistics as st, random, sys
from collections import defaultdict
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from nflmodel import teams
rows = json.load(open("joined_2026.json"))
led = {(r["week"], r["player_name"]): r for r in json.load(open("ledger.json"))["player_snapshots"] if r["status"]=="graded"}
# defense-allowed EPA / success / explosive by opponent, point-in-time (prior 2026 weeks + 2025 shrunk)
def load(season):
    raw = gzip.decompress(open(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "cache" / f"play_by_play_{season}.csv.gz"),"rb").read()).decode("utf-8","replace")
    out=[]
    for r in csv.DictReader(io.StringIO(raw)):
        if r.get("season_type")!="REG" or r.get("play_type") not in ("pass","run") or not r.get("epa"): continue
        out.append((int(r["week"]), teams.canonical(r["defteam"]), r["play_type"], float(r["epa"]), float(r.get("yards_gained") or 0), r.get("success")=="1"))
    return out
p25, p26 = load(2025), load(2026)
def dstats(week):
    acc=defaultdict(lambda: defaultdict(lambda:[0.0,0.0]))
    for src, w in ((p25, 0.35), (p26, 1.0)):
        for wk,d,pt,epa,yds,succ in src:
            if src is p26 and wk>=week: continue
            a=acc[d][pt]; a[0]+=w*epa; a[1]+=w
    return {d:{pt:v[0]/v[1] for pt,v in x.items()} for d,x in acc.items()}
cache={}
X=defaultdict(list)
for r in rows:
    l=led.get((r["week"], r["player"]))
    if not l: continue
    wk=r["week"]; cache.setdefault(wk, dstats(wk)); ds=cache[wk].get(l["opponent"])
    if not ds: continue
    sc=l.get("scheme_context") or {}
    resid=(r["act"]-r["line"]) / max(abs(r["line"]),1)  # relative miss vs line
    X[r["metric"]].append({"y":resid,"gap":(r["proj"]-r["line"])/max(abs(r["line"]),1),
        "def_pass_epa":ds.get("pass",0),"def_rush_epa":ds.get("run",0),
        "tgt_mult":sc.get("target_multiplier",1.0),"pass_eff":sc.get("pass_efficiency_delta",0),
        "rush_eff":sc.get("rush_efficiency_delta",0),"event":r["event"]})
random.seed(2)
def corr_ci(xs, f):
    g=defaultdict(list)
    for x in xs: g[x["event"]].append(x)
    ks=list(g); c=lambda s: st.correlation([x[f] for x in s],[x["y"] for x in s])
    bs=sorted(c([x for k in random.choices(ks,k=len(ks)) for x in g[k]]) for _ in range(400))
    return c(xs), bs[20], bs[380]
for m in ("receiving_yards","receptions","rushing_yards","passing_yards","carries"):
    xs=X[m]; print(m, "n", len(xs))
    for f in ("gap","def_pass_epa","def_rush_epa","tgt_mult","pass_eff","rush_eff"):
        if st.pstdev([x[f] for x in xs])==0: continue
        r_,lo,hi=corr_ci(xs,f); print(f"   corr(actual-line, {f:13s}) = {r_:+.3f}  90% [{lo:+.3f}, {hi:+.3f}]")
