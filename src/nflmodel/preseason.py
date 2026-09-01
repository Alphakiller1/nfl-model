"""What the model knows before a season has produced any evidence.

Week 1 is the hardest week to forecast and the one a dashboard most needs to fill.
There is no current-season form, so every number on that board is carried
forward, and the only honest question is *how much* to carry.

Two quantities are carried, and they are carried the same way:

* **Power ratings** -- an opponent-adjusted solve over the last three completed
  regular seasons, weighted toward the present.
* **Efficiency form** -- the five opponent-adjusted per-play rates in
  `efficiency.py`, adjusted within each season and then averaged across seasons
  on the same weights.

Both are then blended with whatever the current season has produced:

    weight_on_live = games_played / (games_played + K)

`K` is a shrinkage in units of games, selected at **6.0** on 2,383 out-of-sample
games (K = 4 costs 0.010 points of MAE, K = 12 costs 0.048). It says one
September result is worth about a seventh of the prior, and that a team needs six
games before this season outweighs last season. That is slower than instinct
suggests, and it is the single most consequential constant in this module.

A note on how this was nearly mis-measured: the first version of the sweep
reported *identical* MAE for every K from 1 to 12. `live_weight` had `k` as a
default argument, which binds once at import, so reassigning the module constant
did nothing and the sweep silently measured the same model seven times. A flat
curve across a parameter that obviously matters is worth treating as a bug report
rather than a finding.

**Season weights differ between the two quantities, and the direction is the
interesting part.** Ratings are best carried from the most recent season almost
alone (0.85 / 0.15): the sweep improves monotonically with recency, from 10.375
MAE at flat weights to 10.316 at last-season-only, and 0.85/0.15 captures all but
0.0004 of that while not staking a whole prior on one season. Efficiency form
prefers a longer memory (0.70 / 0.20 / 0.10), because a per-play rate is measured
over roughly a thousand plays rather than seventeen results, so an older season
still carries usable signal.
"""

from __future__ import annotations

from . import efficiency, matrix, ratings

# Offset from the most recent completed season -> weight.
RATING_SEASON_WEIGHTS: dict[int, float] = {0: 0.85, 1: 0.15}
FORM_SEASON_WEIGHTS: dict[int, float] = {0: 0.70, 1: 0.20, 2: 0.10}

# Games of current-season evidence needed to match the prior's weight.
BLEND_K = 6.0


def _prior_seasons(season: int, weights: dict[int, float]) -> dict[int, float]:
    """Absolute season -> weight, for the seasons preceding `season`."""
    return {season - 1 - offset: weight for offset, weight in weights.items()}


def live_weight(games_played: float, *, k: float | None = None) -> float:
    """Share of the estimate that comes from the current season.

    `k` resolves against the module constant at call time rather than being
    captured as a default. A default argument is bound once at import, which
    silently turned the tuning sweep for this very parameter into a no-op --
    every K produced identical MAE, which is what a broken sweep looks like.
    """
    if games_played <= 0:
        return 0.0
    k = BLEND_K if k is None else k
    return float(games_played) / (float(games_played) + k)


def rating_prior(history: list[ratings.Game], season: int) -> dict[str, float]:
    """Opponent-adjusted ratings over the completed seasons before `season`."""
    weights = _prior_seasons(season, RATING_SEASON_WEIGHTS)
    scoped = [g for g in history if g.season in weights]
    if not scoped:
        return {}
    # Season weighting is applied inside the solve rather than by averaging three
    # separate solves: the three seasons share opponents, and solving them
    # jointly lets a 2024 result inform a 2023 opponent's rating.
    return ratings.build(scoped, halflife=None, season_weights=weights)


def blend_ratings(prior: dict[str, float], live: dict[str, float],
                  games_played: dict[str, float], *,
                  k: float | None = None) -> dict[str, float]:
    """Prior and current-season ratings, per team, on each team's own game count."""
    out = dict(prior)
    for team, value in live.items():
        weight = live_weight(games_played.get(team, 0.0), k=k)
        out[team] = (1.0 - weight) * prior.get(team, 0.0) + weight * value
    return out


def _adjusted_by_season(lines: list[efficiency.GameLine]
                        ) -> dict[int, dict[str, tuple[dict, dict]]]:
    """Adjust each season independently. Pooling them would let a 2023 defence
    adjust a 2025 offence, which is not a schedule effect."""
    by_season: dict[int, list[efficiency.GameLine]] = {}
    for line in lines:
        by_season.setdefault(line.season, []).append(line)
    return {season: efficiency.adjust_all(rows) for season, rows in by_season.items()}


def form_prior(lines: list[efficiency.GameLine], season: int) -> dict[str, dict[str, float]]:
    """Weighted mean of prior seasons' opponent-adjusted rates, per team.

    Returns ``{team: {"off_epa": ..., "def_epa": ..., ..., "plays": ...}}``.
    """
    weights = _prior_seasons(season, FORM_SEASON_WEIGHTS)
    scoped = [line for line in lines if line.season in weights]
    if not scoped:
        return {}
    adjusted = _adjusted_by_season(scoped)
    pace = {s: efficiency.pace_means([line for line in scoped if line.season == s])
            for s in adjusted}

    accumulated: dict[str, dict[str, float]] = {}
    totals: dict[str, dict[str, float]] = {}
    for season_key, per_stat in adjusted.items():
        weight = weights.get(season_key, 0.0)
        if weight <= 0:
            continue
        for stat, (off, dfn) in per_stat.items():
            for team, value in off.items():
                accumulated.setdefault(team, {})[f"off_{stat}"] = (
                    accumulated.get(team, {}).get(f"off_{stat}", 0.0) + weight * value)
                totals.setdefault(team, {})[f"off_{stat}"] = (
                    totals.get(team, {}).get(f"off_{stat}", 0.0) + weight)
            for team, value in dfn.items():
                accumulated.setdefault(team, {})[f"def_{stat}"] = (
                    accumulated.get(team, {}).get(f"def_{stat}", 0.0) + weight * value)
                totals.setdefault(team, {})[f"def_{stat}"] = (
                    totals.get(team, {}).get(f"def_{stat}", 0.0) + weight)
        for team, values in pace.get(season_key, {}).items():
            accumulated.setdefault(team, {})["plays"] = (
                accumulated.get(team, {}).get("plays", 0.0) + weight * values["plays"])
            totals.setdefault(team, {})["plays"] = (
                totals.get(team, {}).get("plays", 0.0) + weight)

    return {
        team: {key: value / totals[team][key] for key, value in fields.items()
               if totals.get(team, {}).get(key)}
        for team, fields in accumulated.items()
    }


def live_form(lines: list[efficiency.GameLine]) -> dict[str, dict[str, float]]:
    """Opponent-adjusted rates over whatever the current season has produced."""
    if not lines:
        return {}
    adjusted = efficiency.adjust_all(lines)
    pace = efficiency.pace_means(lines)
    out: dict[str, dict[str, float]] = {}
    for stat, (off, dfn) in adjusted.items():
        for team, value in off.items():
            out.setdefault(team, {})[f"off_{stat}"] = value
        for team, value in dfn.items():
            out.setdefault(team, {})[f"def_{stat}"] = value
    for team, values in pace.items():
        out.setdefault(team, {})["plays"] = values["plays"]
    return out


def blend_forms(prior: dict[str, dict[str, float]], live: dict[str, dict[str, float]],
                games_played: dict[str, float], *,
                k: float | None = None) -> dict[str, matrix.TeamForm]:
    """Blend the two form estimates and package them as `TeamForm`.

    A team present in neither is omitted rather than defaulted. A defaulted form
    is indistinguishable from a measured league-average one, and the forecast
    needs to be able to say "no form" out loud.
    """
    out: dict[str, matrix.TeamForm] = {}
    for team in set(prior) | set(live):
        weight = live_weight(games_played.get(team, 0.0), k=k)
        fields: dict[str, float] = {}
        for key in matrix.TeamForm.FIELDS + ("plays",):
            prior_value = prior.get(team, {}).get(key)
            live_value = live.get(team, {}).get(key)
            if prior_value is None and live_value is None:
                continue
            if prior_value is None:
                fields[key] = live_value
            elif live_value is None:
                fields[key] = prior_value
            else:
                fields[key] = (1.0 - weight) * prior_value + weight * live_value
        if fields:
            out[team] = matrix.TeamForm(**fields)
    return out
