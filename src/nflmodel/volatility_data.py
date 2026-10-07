"""NFL inputs for `volatility`: per-game process lines.

Process lines are the same nflverse weekly team box scores the form model is
built from, so the live build reads a file it already has. EPA and turnover
rate are left out on purpose: interceptions and lost fumbles are the luck that
noise cancellation is meant to strip, and EPA carries them at full weight.
"""

from __future__ import annotations

from .efficiency import GameLine

PROCESS_FEATURES = ("first_down", "explosive", "sack", "plays")
# Rates whose game-to-game spread forms the consistency prior.
CONSISTENCY_STATS = ("first_down", "explosive")


def process_index(lines: list[GameLine]) -> dict[tuple[int, int, str, str], dict]:
    """(season, week, team, opponent) -> that team's offensive process line."""
    return {(line.season, line.week, line.team, line.opponent):
            {f: line.value(f) for f in PROCESS_FEATURES}
            for line in lines}
