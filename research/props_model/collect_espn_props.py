"""Collect ESPN DraftKings player O/U prop lines for NFL weeks. python espn_props.py 2026 1,2,3"""
import json, sys, urllib.request, concurrent.futures as cf
UA = {"User-Agent": "Mozilla/5.0"}
def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))

season = int(sys.argv[1]); weeks = [int(w) for w in sys.argv[2].split(",")]
WANT = {"Total Passing Yards (incl. overtime)": "passing_yards",
        "Total Passing Touchdowns (incl. overtime)": "passing_tds",
        "Total Pass Completions (incl. overtime)": "completions",
        "Total Pass Attempts (incl. overtime)": "pass_attempts",
        "Total Passing Interceptions (incl. overtime)": "interceptions",
        "Total Rushing Yards (incl. overtime)": "rushing_yards",
        "Total Carries (incl. overtime)": "carries",
        "Total Receiving Yards (incl. overtime)": "receiving_yards",
        "Total Receptions (incl. overtime)": "receptions"}
out = []
athletes = {}
def event_props(ev):
    eid = ev["id"]
    d = get(f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{eid}/competitions/{eid}/odds/100/propBets?limit=1000")
    rows = []
    seen = set()
    for i in d.get("items", []):
        m = WANT.get(i["type"]["name"])
        if not m or "athlete" not in i or "target" not in i.get("current", {}):
            continue
        aid = i["athlete"]["$ref"].split("/athletes/")[1].split("?")[0]
        k = (aid, m)
        if k in seen:
            continue
        seen.add(k)
        rows.append({"event": eid, "name": ev["name"], "date": ev["date"], "espn_id": aid, "metric": m,
                     "line": i["current"]["target"]["value"],
                     "open": i.get("open", {}).get("target", {}).get("value"),
                     "updated": i.get("lastUpdated")})
    return rows
for w in weeks:
    sb = get(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?seasontype=2&week={w}&dates={season}")
    with cf.ThreadPoolExecutor(8) as ex:
        for rows in ex.map(event_props, sb["events"]):
            for r in rows:
                r["week"] = w
            out.extend(rows)
    print(w, len(sb["events"]), len(out), file=sys.stderr)
json.dump(out, open(f"espn_props_{season}.json", "w"))
