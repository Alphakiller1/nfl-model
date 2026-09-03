"""Immutable pre-kickoff NFL projection ledger and deterministic grader.

Every production build records the exact independent projection and DraftKings
quote it displayed.  A later build grades completed games.  This remains a
shadow record: no stored disagreement is retroactively converted into a bet.
"""

from __future__ import annotations

import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from . import teams

if TYPE_CHECKING:
    from .forecast import GameProjection
    from .player_props import PlayerProjection


SCHEMA_VERSION = "1.1.0"
DEFAULT_PATH = Path(
    os.getenv(
        "NFL_LEDGER_PATH",
        str(
            Path(__file__).resolve().parents[2]
            / "data" / "runtime-cache" / "prediction-ledger.json"
        ),
    )
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(moment: datetime | None = None) -> str:
    return (moment or _now()).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _load(path: Path) -> dict:
    if not path.is_file():
        return {"schema_version": SCHEMA_VERSION, "snapshots": [], "player_snapshots": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload.get("snapshots"), list):
            payload.setdefault("player_snapshots", [])
            return payload
    except (json.JSONDecodeError, AttributeError):
        pass
    return {"schema_version": SCHEMA_VERSION, "snapshots": [], "player_snapshots": []}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _result_index(games: list[dict]) -> dict[tuple[int, int, str, str], dict]:
    out = {}
    for game in games:
        home_score, away_score = game.get("home_score"), game.get("away_score")
        if home_score is None or away_score is None:
            continue
        key = (
            int(game.get("season") or 0),
            int(game.get("week") or 0),
            str(game.get("home_team") or ""),
            str(game.get("away_team") or ""),
        )
        out[key] = game
    return out


def _grade(snapshot: dict, result: dict) -> None:
    actual_margin = float(result["home_score"]) - float(result["away_score"])
    actual_total = float(result["home_score"]) + float(result["away_score"])
    model_margin = snapshot.get("model_margin")
    consensus_margin = snapshot.get("consensus_margin")
    book_margin = snapshot.get("book_margin")
    model_total = snapshot.get("model_total")
    book_total = snapshot.get("book_total")
    snapshot.update({
        "status": "graded",
        "graded_at": _stamp(),
        "actual_margin": actual_margin,
        "actual_total": actual_total,
        "model_abs_error": (
            abs(float(model_margin) - actual_margin) if model_margin is not None else None
        ),
        "consensus_abs_error": (
            abs(float(consensus_margin) - actual_margin) if consensus_margin is not None else None
        ),
        "book_abs_error": (
            abs(float(book_margin) - actual_margin) if book_margin is not None else None
        ),
        "model_total_abs_error": (
            abs(float(model_total) - actual_total) if model_total is not None else None
        ),
        "book_total_abs_error": (
            abs(float(book_total) - actual_total) if book_total is not None else None
        ),
    })
    if model_margin is not None and book_margin is not None:
        direction = 1 if float(model_margin) > float(book_margin) else -1
        result_edge = actual_margin - float(book_margin)
        snapshot["ats_result"] = (
            "push" if abs(result_edge) < 1e-9 else "win" if result_edge * direction > 0 else "loss"
        )
    if model_total is not None and book_total is not None:
        direction = 1 if float(model_total) > float(book_total) else -1
        result_edge = actual_total - float(book_total)
        snapshot["total_result"] = (
            "push" if abs(result_edge) < 1e-9 else "win" if result_edge * direction > 0 else "loss"
        )


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return round(statistics.fmean(values), 4) if values else None


def _player_result_index(rows: list[dict]) -> dict[tuple[int, int, str, str], dict]:
    out = {}
    for row in rows:
        player_id = str(row.get("player_id") or "")
        team = teams.canonical(row.get("team") or "")
        if not player_id or not team:
            continue
        key = (
            int(row.get("season") or 0),
            int(row.get("week") or 0),
            team,
            player_id,
        )
        out[key] = row
    return out


def _num(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _actual_player_metrics(row: dict | None, fields: set[str]) -> dict[str, float]:
    source = row or {}
    mapping = {
        "pass_attempts": "attempts",
        "completions": "completions",
        "passing_yards": "passing_yards",
        "passing_tds": "passing_tds",
        "interceptions": "passing_interceptions",
        "rush_attempts": "carries",
        "carries": "carries",
        "rushing_yards": "rushing_yards",
        "targets": "targets",
        "receptions": "receptions",
        "receiving_yards": "receiving_yards",
        "fg_attempts": "fg_att",
        "fg_made": "fg_made",
        "pat_made": "pat_made",
    }
    actual = {field: _num(source, source_field) for field, source_field in mapping.items()
              if field in fields}
    if "anytime_td_probability" in fields:
        actual["anytime_td_probability"] = float(
            _num(source, "rushing_tds") + _num(source, "receiving_tds") > 0
        )
    if "kicking_points" in fields:
        actual["kicking_points"] = 3.0 * _num(source, "fg_made") + _num(
            source, "pat_made"
        )
    return actual


def _grade_player(snapshot: dict, result: dict | None) -> None:
    projected = snapshot.get("metrics") or {}
    actual = _actual_player_metrics(result, set(projected))
    errors = {
        key: abs(float(value) - actual[key])
        for key, value in projected.items()
        if key in actual
    }
    snapshot.update({
        "status": "graded",
        "graded_at": _stamp(),
        "actual_metrics": actual,
        "absolute_errors": errors,
    })
    if "anytime_td_probability" in projected:
        snapshot["anytime_td_brier"] = (
            float(projected["anytime_td_probability"])
            - actual["anytime_td_probability"]
        ) ** 2


def _player_summary(payload: dict, season: int) -> dict:
    graded = [
        row for row in payload.get("player_snapshots", [])
        if row.get("status") == "graded" and int(row.get("season", 0)) == season
    ]
    latest: dict[tuple, dict] = {}
    for row in graded:
        key = (row["season"], row["week"], row["game_id"], row["player_id"])
        if key not in latest or row.get("recorded_at", "") > latest[key].get(
            "recorded_at", ""
        ):
            latest[key] = row
    rows = list(latest.values())
    errors: dict[str, list[float]] = {}
    for row in rows:
        for metric, value in (row.get("absolute_errors") or {}).items():
            errors.setdefault(metric, []).append(float(value))
    return {
        "scope": "latest pre-kickoff role-aware projection per player-game",
        "authority": "shadow_only",
        "players_graded": len(rows),
        "pending_player_snapshots": sum(
            row.get("status") == "pending"
            for row in payload.get("player_snapshots", [])
        ),
        "mae": {
            metric: round(statistics.fmean(values), 4)
            for metric, values in sorted(errors.items()) if values
        },
        "anytime_td_brier": _mean(rows, "anytime_td_brier"),
    }


def summary(payload: dict, *, season: int) -> dict:
    graded = [
        row for row in payload.get("snapshots", [])
        if row.get("status") == "graded" and int(row.get("season", 0)) == season
    ]
    latest: dict[tuple, dict] = {}
    for row in graded:
        key = (row["season"], row["week"], row["home"], row["away"])
        if key not in latest or row.get("recorded_at", "") > latest[key].get("recorded_at", ""):
            latest[key] = row
    rows = list(latest.values())
    ats = [row.get("ats_result") for row in rows if row.get("ats_result")]
    totals = [row.get("total_result") for row in rows if row.get("total_result")]
    return {
        "scope": "latest pre-kickoff DraftKings snapshot per game",
        "authority": "shadow_only",
        "games_graded": len(rows),
        "pending_snapshots": sum(
            row.get("status") == "pending" for row in payload.get("snapshots", [])
        ),
        "model_mae": _mean(rows, "model_abs_error"),
        "consensus_mae": _mean(rows, "consensus_abs_error"),
        "book_mae": _mean(rows, "book_abs_error"),
        "ats": {name: ats.count(name) for name in ("win", "loss", "push")},
        "model_total_mae": _mean(rows, "model_total_abs_error"),
        "book_total_mae": _mean(rows, "book_total_abs_error"),
        "totals": {name: totals.count(name) for name in ("win", "loss", "push")},
        "players": _player_summary(payload, season),
    }


def update(
    *,
    season: int,
    projections: list["GameProjection"],
    player_projections: list["PlayerProjection"] | None = None,
    player_results: list[dict] | None = None,
    schedule: list[dict],
    path: Path = DEFAULT_PATH,
    recorded_at: datetime | None = None,
) -> dict:
    """Grade pending rows, then append every unseen pre-kickoff book vintage."""
    payload = _load(path)
    results = _result_index(schedule)
    for snapshot in payload["snapshots"]:
        if snapshot.get("status") != "pending":
            continue
        result = results.get(
            (snapshot["season"], snapshot["week"], snapshot["home"], snapshot["away"])
        )
        if result is not None:
            _grade(snapshot, result)

    player_index = _player_result_index(player_results or [])
    for snapshot in payload["player_snapshots"]:
        if snapshot.get("status") != "pending":
            continue
        game_result = results.get(
            (snapshot["season"], snapshot["week"], snapshot["home"], snapshot["away"])
        )
        if game_result is None:
            continue
        result = player_index.get((
            snapshot["season"], snapshot["week"], snapshot["team"], snapshot["player_id"]
        ))
        # Once the team game is final, an absent stat row is a zero-stat DNP,
        # not an indefinitely pending observation.
        _grade_player(snapshot, result)

    now = recorded_at or _now()
    known = {row.get("snapshot_id") for row in payload["snapshots"]}
    for projection in projections:
        kickoff = _parse(projection.kickoff_utc)
        if kickoff is None or kickoff <= now or not projection.book_name:
            continue
        quote_key = projection.book_last_update or _stamp(now)
        snapshot_id = "|".join((
            str(projection.season), str(projection.week), projection.away, projection.home,
            str(projection.book_key), quote_key,
        ))
        if snapshot_id in known:
            continue
        payload["snapshots"].append({
            "snapshot_id": snapshot_id,
            "recorded_at": _stamp(now),
            "season": projection.season,
            "week": projection.week,
            "home": projection.home,
            "away": projection.away,
            "kickoff": projection.kickoff_utc,
            "model_lineage": "2026.09-time-forward-audited-symmetric-matchup",
            "model_margin": projection.model_margin,
            "consensus_margin": projection.market_margin,
            "book": projection.book_name,
            "book_key": projection.book_key,
            "book_margin": projection.book_margin,
            "book_total": projection.book_total,
            "home_moneyline": projection.home_moneyline,
            "away_moneyline": projection.away_moneyline,
            "book_last_update": projection.book_last_update,
            "model_total": projection.projected_total,
            "status": "pending",
            "authority": "shadow_only",
        })
        known.add(snapshot_id)

    known_players = {row.get("snapshot_id") for row in payload["player_snapshots"]}
    for projection in player_projections or []:
        kickoff = _parse(projection.kickoff_utc)
        if kickoff is None or kickoff <= now:
            continue
        snapshot_id = "|".join((
            str(projection.season), str(projection.week), projection.game_id,
            projection.player_id, projection.model_version, _stamp(now),
        ))
        if snapshot_id in known_players:
            continue
        # Home/away identity makes completion grading independent of whether the
        # player produced an official stat row.
        home = projection.team if projection.home else projection.opponent
        away = projection.opponent if projection.home else projection.team
        payload["player_snapshots"].append({
            "snapshot_id": snapshot_id,
            "recorded_at": _stamp(now),
            "season": projection.season,
            "week": projection.week,
            "game_id": projection.game_id,
            "home": home,
            "away": away,
            "kickoff": projection.kickoff_utc,
            "team": projection.team,
            "opponent": projection.opponent,
            "player_id": projection.player_id,
            "player_name": projection.player_name,
            "position": projection.position,
            "depth_rank": projection.depth_rank,
            "injury_status": projection.injury_status,
            "role_continuity": projection.role_continuity,
            "persistence_weight": projection.persistence_weight,
            "confidence": projection.confidence,
            "model_version": projection.model_version,
            "scheme_context": projection.scheme_context,
            "metrics": projection.metrics,
            "status": "pending",
            "authority": "shadow_only",
        })
        known_players.add(snapshot_id)

    payload["schema_version"] = SCHEMA_VERSION
    payload["updated_at"] = _stamp(now)
    payload["snapshots"] = [
        row for row in payload["snapshots"] if int(row.get("season", season)) >= season - 2
    ][-5000:]
    payload["player_snapshots"] = [
        row for row in payload["player_snapshots"]
        if int(row.get("season", season)) >= season - 2
    ][-100000:]
    payload["summary"] = summary(payload, season=season)
    _write(path, payload)
    return payload
