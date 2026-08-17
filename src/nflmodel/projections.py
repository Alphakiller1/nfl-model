"""Consume the versioned preseason outlook published by ``nfl-genesis``.

The public dashboard intentionally has no independent schedule-projection logic. Genesis
creates the artifact, validates its authority and coverage, and this layer fails closed if
the artifact is missing or malformed. This prevents a stale dashboard from quietly drifting
away from the research source of truth.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

OUTLOOK_SCHEMA = "genesis/season-outlook/1"
OUTLOOK_SEASON = 2026
_OUTLOOK_PATH = Path(__file__).resolve().parent / "genesis_outlook_2026.json"


class OutlookError(ValueError):
    """Raised when the Genesis handoff is not safe to render publicly."""


def team_logo_url(team: str) -> str:
    """Public ESPN CDN mark for a known NFL abbreviation."""
    slug = {"LA": "lar"}.get(str(team).upper(), str(team).lower())
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{slug}.png"


def _validate(payload: dict) -> None:
    if payload.get("schema") != OUTLOOK_SCHEMA:
        raise OutlookError("Unsupported Genesis season-outlook schema")
    if payload.get("season") != OUTLOOK_SEASON:
        raise OutlookError(f"Expected outlook for {OUTLOOK_SEASON}")
    if payload.get("authority") != "RESEARCH_ONLY":
        raise OutlookError("A public preseason outlook must be RESEARCH_ONLY")
    games = payload.get("week_one") or []
    divisions = payload.get("division_projections") or []
    if len(games) != 16 or len(divisions) != 8:
        raise OutlookError("Genesis outlook must include 16 games and 8 division projections")
    teams = [game.get(side) for game in games for side in ("away_team", "home_team")]
    if len(set(teams)) != 32 or any(not team for team in teams):
        raise OutlookError("Genesis Week 1 outlook must cover all 32 teams exactly once")
    for game in games:
        try:
            probability = float(game["home_win_probability"]) + float(
                game["away_win_probability"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise OutlookError(f"Invalid game probability: {game.get('game')}") from exc
        if not math.isclose(probability, 1.0, abs_tol=1e-4):
            raise OutlookError(f"Game probabilities do not sum to one: {game.get('game')}")


@lru_cache(maxsize=1)
def outlook() -> dict:
    """Load the checked-in Genesis artifact; never synthesize a fallback forecast."""
    try:
        payload = json.loads(_OUTLOOK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OutlookError("Genesis outlook artifact is unavailable") from exc
    _validate(payload)
    return payload


def week_one_projections() -> list[dict]:
    return list(outlook()["week_one"])


def division_winners() -> list[dict]:
    return list(outlook()["division_projections"])


def schedule_source() -> str:
    source = outlook().get("source") or {}
    return str(source.get("week_one_schedule") or "")


def outlook_note() -> str:
    return str(outlook().get("note") or "")
