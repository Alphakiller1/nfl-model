"""
Preseason power ratings — what the model believes before a market exists.

The game forecast in `forecast.py` is market-anchored, so with `lam = 0` it *equals* the
paired no-vig closing line by construction. That is the honest answer in season, but out of
season there is no line to anchor to, so the board had nothing to render and the product
looked empty. A power rating is a different and weaker claim than an edge, and stating it
plainly is better than shipping a blank page: it says what the model thinks team strength
is, without pretending that belief beats anybody's price.

Ratings are computed in `nfl-genesis` (which owns the historical data) by
`scripts/build_power_ratings.py` and vendored here as JSON so this package keeps its
zero-dependency guarantee — no pandas, no parquet reader, just `json` from the stdlib.

A rating is points relative to an average team on a neutral field, so the projected neutral
margin between two teams is the difference of their ratings, and home field adds
`home_field_points` to the host.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

_RATINGS_PATH = Path(__file__).resolve().parent / "power_ratings.json"

# The rating export uses ``LA`` while the official schedule uses ``LAR``. Normalize
# at the model boundary so schedule-driven projections cannot silently omit the Rams.
_TEAM_ALIASES = {"LAR": "LA"}

# Standard deviation of an NFL game margin. Stable near 13.5 points across modern seasons
# and used only to turn a rating gap into a win probability; the projection below is not
# sensitive to small changes in it.
MARGIN_SD = 13.5
GAMES_PER_SEASON = 17


@lru_cache(maxsize=1)
def load() -> dict:
    """The vendored ratings payload, or an empty shell when it has not been built."""
    if not _RATINGS_PATH.is_file():
        return {"teams": [], "seasons": [], "home_field_points": 0.0}
    return json.loads(_RATINGS_PATH.read_text(encoding="utf-8"))


def teams() -> list[dict]:
    return list(load().get("teams") or [])


def rating_for(team: str) -> float | None:
    raw_key = str(team or "").upper()
    key = _TEAM_ALIASES.get(raw_key, raw_key)
    for entry in teams():
        if str(entry.get("team", "")).upper() == key:
            return float(entry.get("rating", 0.0))
    return None


def projected_margin(home: str, away: str, *, neutral: bool = False) -> float | None:
    """Projected home margin in points. `None` when either team is unrated."""
    home_rating, away_rating = rating_for(home), rating_for(away)
    if home_rating is None or away_rating is None:
        return None
    margin = home_rating - away_rating
    if not neutral:
        margin += float(load().get("home_field_points") or 0.0)
    return round(margin, 2)


def win_probability(margin: float) -> float:
    """P(win) implied by a projected point margin, on the game-margin scale."""
    return 0.5 * (1.0 + math.erf(float(margin) / (MARGIN_SD * math.sqrt(2.0))))


def projected_wins(team: str) -> float | None:
    """Expected regular-season wins against a **league-average schedule**.

    Deliberately schedule-neutral: this repo vendors ratings, not a fixture list, and the
    2026 schedule is not published in the research repo yet. Inventing opponents would make
    the figure look more precise than the inputs support, so this answers a narrower
    question honestly — how many games would this team win playing an average opponent every
    week, split half home and half away, so home field enters once rather than twice.

    A prior on team strength expressed in wins: useful for reading a season win total,
    explicitly not a priced edge against one.
    """
    rating = rating_for(team)
    if rating is None:
        return None
    hfa = float(load().get("home_field_points") or 0.0)
    home = win_probability(rating + hfa)
    away = win_probability(rating - hfa)
    return round(GAMES_PER_SEASON / 2 * (home + away), 1)


def win_projection_table() -> list[dict]:
    """Every rated team with its projected wins, ordered strongest first."""
    rows = []
    for entry in teams():
        team = str(entry.get("team", ""))
        rows.append({
            "team": team,
            "rating": float(entry.get("rating", 0.0)),
            "rank": entry.get("rank"),
            "projected_wins": projected_wins(team),
            "last_season": entry.get("last_season") or {},
        })
    rows.sort(key=lambda row: -(row["projected_wins"] or 0.0))
    return rows
