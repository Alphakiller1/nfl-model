"""Fail a Pages release whose manifest cannot prove a usable current slate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def verify(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for key in ("generated_at", "season", "week", "nflverse", "odds", "players"):
        if key not in payload:
            errors.append(f"missing manifest field: {key}")

    odds = payload.get("odds") or {}
    games = int(odds.get("slate_games") or 0)
    matched = int(odds.get("slate_matched") or 0)
    complete = int(odds.get("slate_complete") or 0)
    if games <= 0:
        errors.append("current week contains no regular-season games")
    if games and matched <= 0:
        errors.append("no DraftKings lines matched the slate")
    minimum = float(os.getenv("NFL_MIN_BOOK_COVERAGE", "1.00"))
    if games and complete / games < minimum:
        errors.append(
            f"complete DraftKings coverage {complete}/{games} is below {minimum:.0%}"
        )
    for field, label in (
        ("slate_spreads", "spreads"),
        ("slate_totals", "totals"),
        ("slate_moneylines", "paired moneylines"),
    ):
        count = int(odds.get(field) or 0)
        if games and count / games < minimum:
            errors.append(f"DraftKings {label} coverage {count}/{games} is below {minimum:.0%}")
    if str(odds.get("requested_book") or "").lower() != "draftkings":
        errors.append("production sportsbook is not exactly draftkings")
    if odds.get("state") not in {"fresh", "cached"}:
        errors.append(f"sportsbook source state is {odds.get('state')!r}")
    age = odds.get("age_seconds")
    maximum_age = int(os.getenv("NFL_MAX_ODDS_AGE_SECONDS", "1200"))
    if age is None:
        errors.append("sportsbook snapshot age is unavailable")
    elif int(age) > maximum_age:
        errors.append(f"sportsbook snapshot is {age}s old (maximum {maximum_age}s)")

    for issue in payload.get("issues", []):
        errors.append(f"publication issue: {issue}")

    if os.getenv("NFL_REQUIRE_FRESH_NFLVERSE", "1").lower() in {"1", "true", "yes"}:
        stale = [row.get("cache_name") for row in payload.get("nflverse", []) if row.get("stale")]
        failed = [
            row.get("cache_name") for row in payload.get("nflverse", [])
            if row.get("state") == "error"
        ]
        if stale:
            errors.append(f"{len(stale)} nflverse source(s) are stale")
        if failed:
            errors.append(f"{len(failed)} nflverse source(s) failed")

    players = payload.get("players") or {}
    player_count = int(players.get("players_projected") or 0)
    team_count = int(players.get("teams_covered") or 0)
    if player_count <= 0:
        errors.append("no offensive player or kicker projections were generated")
    if team_count < 32:
        errors.append(f"player projections cover {team_count}/32 teams")
    if int(players.get("active_roster_week") or 0) < int(payload.get("week") or 0):
        errors.append("active roster snapshot predates the projected week")
    if not players.get("depth_chart_as_of"):
        errors.append("depth-chart timestamp is unavailable")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = verify(args.manifest)
    odds = payload.get("odds") or {}
    matched = int(odds.get("slate_matched") or 0)
    complete = int(odds.get("slate_complete") or 0)
    print(
        "sportsbook diagnostic: "
        f"book={odds.get('requested_book')} state={odds.get('state')} "
        f"feed={odds.get('matched')}/{odds.get('events')} "
        f"slate_events={matched}/{odds.get('slate_games')} "
        f"complete={complete}/{odds.get('slate_games')} "
        f"credits_remaining={odds.get('remaining')}"
    )
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print(
        f"verified {payload['season']} week {payload['week']}: "
        f"{odds['slate_complete']}/{odds['slate_games']} complete DraftKings line sets"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
