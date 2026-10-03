"""Point-in-time replay of player_props.project over a past season.

Writes one row per projected player-week with the projection, the actual, and
team-level context, to replay_<season>.json. Usage:
    python replay.py 2025 [weeks]
"""
import json, os, sys
from types import SimpleNamespace
from collections import defaultdict

sys.path.insert(0, os.environ.get("NFLMODEL_SRC", str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src")))
from nflmodel import player_props, teams  # noqa
from nflmodel.sources import nflverse  # noqa
from nflmodel.season import kickoff_utc  # noqa
exec(os.environ.get('PATCH', ''))

season = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
weeks = range(2, 19) if len(sys.argv) < 3 else [int(w) for w in sys.argv[2].split(",")]
out_path = sys.argv[3] if len(sys.argv) > 3 else f"replay_{season}.json"

schedule = nflverse.games(seasons=(season,))
history_all = []
for s in (season - 2, season - 1):
    history_all.extend(nflverse.player_week(s, completed_season=True))
current = nflverse.player_week(season, completed_season=True)
SNAPS = []
if os.environ.get('SNAPS', '1') == '1':
    for s_ in (season - 1, season):
        SNAPS.extend(nflverse.snap_counts(s_, completed_season=True))

actual = {}
published = set()
for r in current:
    if str(r.get("season_type") or "REG").upper() != "REG":
        continue
    key = (int(nflverse.number(r["week"])), teams.canonical(r["team"]))
    published.add(key)
    actual[(key[0], key[1], r["player_id"])] = r

METRIC_FIELDS = {
    "pass_attempts": "attempts", "completions": "completions", "passing_yards": "passing_yards",
    "passing_tds": "passing_tds", "interceptions": "passing_interceptions",
    "rush_attempts": "carries", "rushing_yards": "rushing_yards", "carries": "carries",
    "targets": "targets", "receptions": "receptions", "receiving_yards": "receiving_yards",
    "fg_attempts": "fg_att", "fg_made": "fg_made", "pat_made": "pat_made",
}

team_actual = defaultdict(lambda: defaultdict(float))
for r in current:
    if str(r.get("season_type") or "REG").upper() != "REG":
        continue
    k = (int(nflverse.number(r["week"])), teams.canonical(r["team"]))
    for f in ("attempts", "carries", "targets", "passing_yards", "rushing_yards", "receiving_yards"):
        team_actual[k][f] += float(nflverse.number(r.get(f)) or 0)

rows = []
for week in weeks:
    games = [g for g in schedule if g["season"] == season and g["week"] == week and g["game_type"] == "REG"]
    if not games:
        continue
    first = min(kickoff_utc(g) for g in games if kickoff_utc(g))
    roster = nflverse.weekly_roster(season, week=week)
    depth = nflverse.depth_charts(season, before=first)
    injuries = nflverse.injuries(season, week=week)
    history = history_all + [r for r in current if int(nflverse.number(r.get("week")) or 0) < week]
    gps = []
    for g in games:
        gps.append(SimpleNamespace(
            home=teams.canonical(g["home_team"]), away=teams.canonical(g["away_team"]),
            book_total=nflverse.number(g.get("total_line")),
            book_margin=nflverse.number(g.get("spread_line")),
            kickoff="", kickoff_utc=kickoff_utc(g)))
    res = player_props.project(season=season, week=week, games=games, game_projections=gps,
                               roster=roster, depth=depth, injuries=injuries, history_rows=history, snap_rows=SNAPS)
    for p in res.projections:
        if (week, p.team) not in published:
            continue
        a = actual.get((week, p.team, p.player_id))
        act = {}
        for m in p.metrics:
            if m == "anytime_td_probability":
                act[m] = 0.0 if a is None else float(
                    (nflverse.number(a.get("rushing_tds")) or 0) + (nflverse.number(a.get("receiving_tds")) or 0) > 0)
            elif m == "kicking_points":
                act[m] = 0.0 if a is None else 3 * float(nflverse.number(a.get("fg_made")) or 0) + float(nflverse.number(a.get("pat_made")) or 0)
            else:
                act[m] = 0.0 if a is None else float(nflverse.number(a.get(METRIC_FIELDS[m])) or 0)
        ta = team_actual[(week, p.team)]
        rows.append({
            "week": week, "team": p.team, "opp": p.opponent, "home": p.home,
            "pid": p.player_id, "name": p.player_name, "pos": p.position, "rank": p.depth_rank,
            "inj": p.injury_status, "cont": p.role_continuity, "hist": p.history_games,
            "implied": p.implied_team_points, "played": a is not None,
            "m": p.metrics, "a": act, "team_act": dict(ta),
        })
    print(week, len(res.projections), file=sys.stderr)

json.dump(rows, open(out_path, "w"))
print("rows", len(rows))
