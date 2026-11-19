"""Reproduce structural player forecasts and strictly pre-week process features.

Research dependencies: numpy/pandas for fitting; production stays standard-library
only. Historical game lines are closing benchmarks, not timestamped entry quotes.
Weekly historical rosters/injuries are retrospective records, not vintage replay.
Participation from the forecast season is excluded even when it exists today.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nflmodel import player_props, prop_intelligence, scheme, teams  # noqa: E402
from nflmodel.season import kickoff_utc  # noqa: E402
from nflmodel.sources import nflverse  # noqa: E402

METRICS = {
    "pass_attempts": "attempts",
    "completions": "completions",
    "passing_yards": "passing_yards",
    "passing_tds": "passing_tds",
    "interceptions": "passing_interceptions",
    "rush_attempts": "carries",
    "rushing_yards": "rushing_yards",
    "carries": "carries",
    "targets": "targets",
    "receptions": "receptions",
    "receiving_yards": "receiving_yards",
    "fg_attempts": "fg_att",
    "fg_made": "fg_made",
    "pat_made": "pat_made",
}


def read_csv(name: str) -> list[dict]:
    with (nflverse.CACHE_DIR / name).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def select_depth(rows: list[dict], *, week: int, cutoff: str) -> list[dict]:
    if rows and "dt" in rows[0]:
        latest = max((r["dt"] for r in rows if r["dt"] <= cutoff), default="")
        return [r for r in rows if r["dt"] == latest] if latest else []
    # The 2024 provider used week-labelled depth records, without an observation
    # timestamp. Do not give those records invented timestamps.
    return [
        {
            "team": r.get("club_code"),
            "gsis_id": r.get("gsis_id"),
            "player_name": r.get("full_name"),
            "pos_abb": r.get("position"),
            "pos_slot": r.get("depth_position"),
            "pos_rank": r.get("depth_team"),
        }
        for r in rows
        if int(nflverse.number(r.get("week")) or 0) == week
        and str(r.get("game_type") or "REG") == "REG"
    ]


def select_roster(rows: list[dict], week: int) -> list[dict]:
    latest = defaultdict(int)
    for row in rows:
        candidate = int(nflverse.number(row.get("week")) or 0)
        if candidate <= week:
            latest[row["team"]] = max(latest[row["team"]], candidate)
    return [r for r in rows if int(nflverse.number(r.get("week")) or 0) == latest[r["team"]]]


def write_gzip(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    path.write_bytes(gzip.compress(raw, mtime=0))


def replay(
    season: int, out: Path, *, weeks: list[int] | None = None, features_only: bool = False
) -> None:
    schedule = nflverse.games(ttl=None)
    current = nflverse.player_week(season)
    history_all = []
    for year in range(season - 3, season):
        history_all.extend(nflverse.player_week(year, completed_season=True))
    roster_all = nflverse.fetch_csv(
        nflverse.WEEKLY_ROSTER_URL.format(season=season), f"roster_weekly_{season}.csv", ttl=None
    )
    snaps = nflverse.snap_counts(season - 1) + nflverse.snap_counts(season)
    if not (nflverse.CACHE_DIR / f"depth_charts_{season}.csv").exists():
        nflverse.depth_charts(season)
    depth_all = read_csv(f"depth_charts_{season}.csv")
    injuries = nflverse.fetch_csv(
        nflverse.INJURIES_URL.format(season=season), f"injuries_{season}.csv", ttl=None
    )
    pbp = nflverse.play_by_play(season - 1) + nflverse.play_by_play(season)
    part = nflverse.participation(season - 1)
    chart = nflverse.ftn_charting(season - 1) + nflverse.ftn_charting(season)
    print(
        f"{season}: loaded {len(current)} player rows, {len(pbp)} plays, "
        f"{len(part)} prior-season charted plays",
        flush=True,
    )
    actual = {
        (int(nflverse.number(r["week"]) or 0), teams.canonical(r["team"]), r["player_id"]): r
        for r in current
        if str(r.get("season_type") or "REG") == "REG"
    }
    completed = [
        g
        for g in schedule
        if g["season"] == season
        and g["game_type"] == "REG"
        and g["home_score"] is not None
        and g["away_score"] is not None
    ]
    all_weeks = sorted({g["week"] for g in completed if g["week"] >= 2})
    selected_weeks = [w for w in all_weeks if weeks is None or w in weeks]
    output = []
    saved = (
        json.loads(gzip.decompress((out / f"structural-{season}.json.gz").read_bytes()))
        if features_only
        else []
    )
    for week in selected_weeks:
        started = time.monotonic()
        games = [g for g in completed if g["week"] == week]
        first = min(kickoff_utc(g) for g in games if kickoff_utc(g))
        roster = select_roster(roster_all, week)
        depth = select_depth(depth_all, week=week, cutoff=first)
        history = history_all + [
            r for r in current if int(nflverse.number(r.get("week")) or 0) < week
        ]
        injury_week = [r for r in injuries if int(nflverse.number(r.get("week")) or 0) == week]
        game_projections = [
            SimpleNamespace(
                home=teams.canonical(g["home_team"]),
                away=teams.canonical(g["away_team"]),
                book_total=nflverse.number(g.get("total_line")),
                book_margin=nflverse.number(g.get("spread_line")),
                kickoff="",
                kickoff_utc=kickoff_utc(g),
            )
            for g in games
        ]
        schematic = scheme.build(
            season=season,
            week=week,
            games=games,
            schedule=schedule,
            pbp_rows=pbp,
            participation_rows=part,
            charting_rows=chart,
            player_positions=scheme.position_index(history),
        )
        if features_only:
            forecasts = {(p.home, p.away): p for p in game_projections}
            players = []
            for row in saved:
                if row["week"] != week:
                    continue
                game = next(g for g in games if g["game_id"] == row["game_id"])
                home, away = teams.canonical(game["home_team"]), teams.canonical(game["away_team"])
                forecast = forecasts[(home, away)]
                margin = float(forecast.book_margin or 0) * (1 if row["team"] == home else -1)
                implied = (float(forecast.book_total or 44.5) + margin) / 2
                players.append(SimpleNamespace(**row, implied_team_points=implied))
        else:
            build = player_props.project(
                season=season,
                week=week,
                games=games,
                game_projections=game_projections,
                roster=roster,
                depth=depth,
                injuries=injury_week,
                history_rows=history,
                snap_rows=snaps,
                scheme_matchups=schematic.matchups,
                box_rates=player_props.defense_box_rates(pbp, chart, season=season, week=week),
                intelligence_enabled=False,
            )
            players = build.projections
        processes = prop_intelligence.process_games(
            pbp,
            season=season,
            week=week,
            participation_rows=part,
            charting_rows=chart,
        )
        histories = prop_intelligence.history_index(history, season=season, week=week)
        totals = prop_intelligence.team_totals(history, season=season, week=week)
        for player in players:
            actual_row = actual.get((week, player.team, player.player_id)) or {}
            outcomes = {
                metric: float(nflverse.number(actual_row.get(field)) or 0)
                for metric, field in METRICS.items()
                if metric in player.metrics
            }
            if "anytime_td_probability" in player.metrics:
                outcomes["anytime_td_probability"] = float(
                    sum(
                        float(nflverse.number(actual_row.get(key)) or 0)
                        for key in ("receiving_tds", "rushing_tds")
                    )
                    > 0
                )
            if "kicking_points" in player.metrics:
                outcomes["kicking_points"] = 3 * outcomes["fg_made"] + outcomes["pat_made"]
            features, evidence = prop_intelligence.feature_profile(
                player,
                histories,
                processes,
                matchup=schematic.matchups.get((player.team, player.opponent)),
                team_totals=totals,
            )
            output.append(
                {
                    "season": season,
                    "week": week,
                    "player_id": player.player_id,
                    "name": getattr(player, "player_name", getattr(player, "name", "")),
                    "position": player.position,
                    "team": player.team,
                    "opponent": player.opponent,
                    "game_id": player.game_id,
                    "played": bool(actual_row),
                    "depth_rank": player.depth_rank,
                    "injury_status": player.injury_status,
                    "metrics": player.metrics,
                    "actual": outcomes,
                    "features": features,
                    "evidence": evidence,
                }
            )
        print(
            f"{season} week {week}: {len(players)} players in {time.monotonic() - started:.1f}s",
            flush=True,
        )
        write_gzip(out / f"structural-{season}.json.gz", output)
    receipts = []
    for status in nflverse.status_report():
        path = nflverse.CACHE_DIR / status["cache_name"]
        receipts.append(
            {
                **status,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
            }
        )
    (out / f"sources-{season}.json").write_text(json.dumps(receipts, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seasons", default="2024,2025,2026")
    parser.add_argument("--weeks")
    parser.add_argument("--features-only", action="store_true")
    args = parser.parse_args()
    nflverse.CACHE_DIR = args.cache.resolve()
    os.environ["NFL_MODEL_CACHE"] = str(nflverse.CACHE_DIR)
    weeks = [int(w) for w in args.weeks.split(",")] if args.weeks else None
    for season in [int(s) for s in args.seasons.split(",")]:
        replay(season, args.out, weeks=weeks, features_only=args.features_only)


if __name__ == "__main__":
    main()
