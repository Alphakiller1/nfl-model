"""Read the observed MLBMA NFL scouting snapshot, with a pre-forecast cutoff.

This is a read-only consumer. Neither model forecasts in MLBMA nor its public
rankings are fed back into the projection matrix. Unavailable or mismatched
snapshots leave an explicit status and the local scheme evidence still works.
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

URL = "https://chase-analytics.com/data/public/nfl/slate.json"
MAX_AGE_HOURS = 72
MAX_BYTES = 12_000_000


def timestamp(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(UTC) if parsed.tzinfo else None
    except (ValueError, TypeError):
        return None


def _finite(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    return True


def _valid_measures(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, (float, int)) and not isinstance(child, bool):
                if key.endswith(("_rate", "_share")) and not 0 <= child <= 1:
                    return False
                if key in {"targets", "carries", "attempts", "dropbacks", "receptions", "plays",
                           "games", "trips", "td_trips", "score_trips"} and child < 0:
                    return False
            if not _valid_measures(child):
                return False
    elif isinstance(value, list):
        return all(_valid_measures(child) for child in value)
    return True


def validate(payload: dict, *, season: int, week: int, now: datetime, source: str = URL) -> dict:
    """Reject future, stale, wrong-week and undated per-game evidence.

    The public slate's top-level data_through is the schedule's last game;
    scheme_source.observed_at_utc dates its actual scouting evidence. Both
    that evidence and the snapshot must exist by ``now`` and before kickoff.
    A historical replay must inject the snapshot that existed at its cutoff.
    """
    result = {"state": "unavailable", "source": source, "games": [], "issues": []}
    if not isinstance(payload, dict) or not _finite(payload):
        result["issues"] = ["Invalid or non-finite scouting snapshot"]
        return result
    generated = timestamp(payload.get("generated_at_utc"))
    if payload.get("schema") != "chase-public-slate/1" or payload.get("sport") != "nfl":
        result["issues"] = ["Unrecognized scouting snapshot contract"]
        return result
    if generated is None or generated > now:
        result["issues"] = ["Scouting snapshot is undated or later than the forecast"]
        return result
    if (now - generated).total_seconds() > MAX_AGE_HOURS * 3600:
        result["issues"] = ["Scouting snapshot exceeds the 72-hour freshness limit"]
        return result
    games = payload.get("games")
    if not isinstance(games, list):
        result["issues"] = ["Scouting snapshot has no game list"]
        return result
    for game in games:
        if not isinstance(game, dict):
            continue
        provenance = game.get("scheme_source") or {}
        observed = timestamp(provenance.get("observed_at_utc"))
        kickoff = timestamp(game.get("kickoff_utc"))
        if provenance.get("season") != season or provenance.get("week") != week:
            continue
        if observed is None or observed > generated or observed > now:
            continue
        if (now - observed).total_seconds() > MAX_AGE_HOURS * 3600:
            continue
        if kickoff is None or kickoff <= now or observed >= kickoff:
            continue
        if not _valid_measures(game):
            result["issues"].append("Rejected a fixture with invalid observed rates or counts")
            continue
        result["games"].append(game)
    result.update(
        {
            "state": "fresh" if result["games"] else "unavailable",
            "generated_at_utc": generated.isoformat(),
        }
    )
    if not result["games"]:
        result["issues"] = ["No dated pre-kickoff scouting evidence for this season/week"]
    return result


def load(*, season: int, week: int, now: datetime | None = None) -> dict:
    moment = now or datetime.now(UTC)
    local = os.getenv("NFL_CHASE_CONTEXT_PATH")
    source = local or URL
    try:
        if local:
            path = Path(local)
            if path.stat().st_size > MAX_BYTES:
                raise ValueError("Scouting snapshot exceeds size limit")
            raw = path.read_bytes()
        else:
            request = urllib.request.Request(URL, headers={"User-Agent": "nfl-model/weekly-props"})
            with urllib.request.urlopen(request, timeout=10) as response:
                raw = response.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise ValueError("Scouting snapshot exceeds size limit")
        return validate(json.loads(raw), season=season, week=week, now=moment, source=source)
    except (OSError, ValueError, TypeError) as exc:
        return {
            "state": "unavailable",
            "source": source,
            "games": [],
            "issues": [f"Scouting snapshot unavailable: {type(exc).__name__}"],
        }
