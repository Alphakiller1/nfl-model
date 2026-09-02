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

if TYPE_CHECKING:
    from .forecast import GameProjection


SCHEMA_VERSION = "1.0.0"
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
        return {"schema_version": SCHEMA_VERSION, "snapshots": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload.get("snapshots"), list):
            return payload
    except (json.JSONDecodeError, AttributeError):
        pass
    return {"schema_version": SCHEMA_VERSION, "snapshots": []}


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
    }


def update(
    *,
    season: int,
    projections: list["GameProjection"],
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

    payload["schema_version"] = SCHEMA_VERSION
    payload["updated_at"] = _stamp(now)
    payload["snapshots"] = [
        row for row in payload["snapshots"] if int(row.get("season", season)) >= season - 2
    ][-5000:]
    payload["summary"] = summary(payload, season=season)
    _write(path, payload)
    return payload
