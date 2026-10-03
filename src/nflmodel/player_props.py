"""Causal, role-aware NFL offensive player and kicker projections.

This is a projection layer, not a player-prop betting model.  It does not ingest
player prices and it never emits an edge.  The current active roster and latest
dated depth chart determine *who has the next-game role*; past production only
estimates the size and efficiency of that role.  That distinction matters most
for rookies, transactions and depth-chart changes, where carrying a trailing
average forward literally would be an inductive but non-predictive mistake.

Every opportunity forecast reconciles to its team environment.  Receiver target
shares sum to one team target pool and running-back carry shares sum to the
backfield pool, so the page cannot project five independent WR1 workloads or
thirty-eight team pass attempts plus twenty-nine player targets.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

from . import teams
from .sources.nflverse import number

MODEL_VERSION = "nfl-player-projections/2.0.0"
POSITIONS = ("QB", "RB", "WR", "TE", "K")
DEPTH_LIMITS = {"QB": 1, "RB": 3, "WR": 4, "TE": 2, "K": 1}
# Share of pre-kickoff depth-chart players (not listed Out) who recorded a carry,
# target or catch: nflverse 2025 depth charts snapped before each week's first
# kickoff, joined to weekly stats (~550 player-weeks per slot). It now scales a
# player's claim on the pool *before* the pool is divided, so the volume a
# likely inactive slot gives up goes to his teammates instead of vanishing
# (applied after the split it removed ~3 carries a game from every backfield).
ACTIVE_RATE = {
    ("RB", 1): 0.923, ("RB", 2): 0.872, ("RB", 3): 0.468,
    ("WR", 1): 0.901, ("WR", 2): 0.884, ("WR", 3): 0.829, ("WR", 4): 0.638,
    ("TE", 1): 0.891, ("TE", 2): 0.686,
}
# Efficiency rates (yards per target, TD rate, ...) keep a long memory; they are
# mostly noise and are shrunk hard (RATE_PRIORS). Usage shares move faster:
# a 6-week half-life minimised next-game share error on 2023-2025 pbp.
HALF_LIFE_WEEKS = 12.0
USAGE_HALF_LIFE_WEEKS = 6.0
# Weighted same-team games at which observed usage and the depth-slot prior
# carry equal weight. Fitted on the 2025 point-in-time replay.
USAGE_PSEUDO_GAMES = 1.0

# Team volume (reports/PROPS_MODEL.md, part A). Team tendencies shift quickly: a
# 4-week half-life with a prior of last season's deviation regressed toward the
# league beat every longer memory on 2022-2025. Defensive "allowed" volume is
# 3-4x noisier than offensive volume, hence its heavier prior.
TEAM_HALF_LIFE_WEEKS = 4.0
TEAM_PRIOR = {  # (pseudo games, share of last season's deviation carried over)
    ("offense", "attempts"): (4.0, 0.5),
    ("offense", "carries"): (6.0, 0.3),
    ("defense", "attempts"): (10.0, 0.3),
    ("defense", "carries"): (10.0, 0.3),
}
# Deviation-form OLS on 2022-2025 team-games (pass attempts excl. sacks;
# designed runs). Favourites run more plays at a lower dropback rate and the two
# cancel, so the spread moves runs, not passes; the total moves passes.
VOLUME_MODEL = {
    "attempts": {"offense": 0.905, "defense": 0.585, "spread": 0.021, "total": 0.175},
    "carries": {"offense": 0.924, "defense": 0.738, "spread": 0.103, "total": -0.102},
}
LEAGUE_TOTAL = 44.5
# Team targets go to position groups first, then to players inside the group,
# so an absent WR's targets go mostly to the other WRs (part B3). Each group
# also keeps a reserve for players outside the projected depth slots (WR5,
# TE3, FB, gadget), measured on the 2025 replay.
GROUP_TARGET_SHARE = {"WR": 0.576, "TE": 0.240, "RB": 0.180}
GROUP_PSEUDO_GAMES = 6.0
GROUP_RESERVE = {"WR": 0.076, "TE": 0.128, "RB": 0.045}
CARRY_RESERVE = 0.035
# Snap trend: observed share x (last game snap % / weighted snap %) ** beta.
# Fitted on 2021-2024, scored on 2025: share MAE -0.5% to -1.7% (part B2).
SNAP_TREND_BETA = {"targets": 0.3, "carries": 0.4}
# WR and TE compete for one pool (on the 2025 replay a separate pool for each
# lost to a shared one); backs keep their own, which won for RB targets and TDs.
TARGET_GROUPS = (("WR", "TE"), ("RB",))
# The depth-chart starter's share of team pass attempts (2025 replay: 0.954);
# injuries, benchings and garbage-time backups take the rest.
QB_ATTEMPT_SHARE = 0.954

# Forward role priors by present-day depth rank.  These are starting points,
# not fixed allocations: established same-team usage can move them materially,
# while new-team and rookie history is deliberately prevented from doing so.
TARGET_PRIORS = {
    "RB": (0.120, 0.065, 0.030),
    "WR": (0.235, 0.185, 0.130, 0.080),
    "TE": (0.145, 0.070),
}
CARRY_PRIORS = {"RB": (0.540, 0.270, 0.120)}

# (league mean, pseudo-count in opportunities). Quarterback pseudo-counts come
# from split-half reliability on 2021-2025 pbp, k = n_half * (1 - r) / r: the
# attempts at which a passer's own rate earns half weight. Interception rate has
# no repeatable signal (r = -0.03). The same estimate for receivers and backs
# (k ~ 120-570) lost to these lighter priors on the 2025 replay, because a
# multi-season weighted history carries more talent signal than one season's
# split halves, so the skill-position values stay where they were.
RATE_PRIORS = {
    "completion": (0.645, 440.0),
    "pass_yards": (6.95, 190.0),
    "pass_td": (0.044, 540.0),
    "interception": (0.023, 5000.0),
    "rush_yards_qb": (4.80, 24.0),
    "rush_yards_rb": (4.30, 45.0),
    "catch_rb": (0.735, 28.0),
    "catch_wr": (0.625, 34.0),
    "catch_te": (0.675, 30.0),
    "receive_yards_rb": (6.20, 30.0),
    "receive_yards_wr": (7.90, 38.0),
    "receive_yards_te": (7.40, 34.0),
    "rush_td_qb": (0.012, 45.0),
    "rush_td_rb": (0.026, 55.0),
    "receive_td_rb": (0.012, 45.0),
    "receive_td_wr": (0.028, 55.0),
    "receive_td_te": (0.031, 50.0),
}
# A receiver's depth of target (aDOT) is far more stable than his yards per
# target (split-half k = 22 targets vs ~120), so the prior his efficiency is
# shrunk toward is read off his aDOT: weighted fits on 2021-2025 seasons.
ADOT_PRIOR = {  # position: (league aDOT, ypt intercept, ypt slope, catch intercept, catch slope)
    "WR": (10.66, 6.08, 0.170, 0.807, -0.0172),
    "TE": (6.80, 5.633, 0.273, 0.810, -0.0145),
    "RB": (0.10, 5.824, 0.228, 0.787, -0.0163),
}
ADOT_PSEUDO_TARGETS = 22.0


@dataclass(frozen=True)
class PlayerProjection:
    season: int
    week: int
    game_id: str
    kickoff: str
    kickoff_utc: str
    team: str
    opponent: str
    home: bool
    player_id: str
    player_name: str
    position: str
    depth_rank: int
    depth_slot: str
    roster_status: str
    injury_status: str | None
    headshot_url: str
    history_games: int
    last_team: str | None
    role_continuity: str
    persistence_weight: float
    role_reason: str
    confidence: str
    team_environment_source: str
    implied_team_points: float | None
    scheme_context: dict[str, object] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    model_version: str = MODEL_VERSION
    authority: str = "RESEARCH_ONLY"
    action: str = "NO BET - projections only"


@dataclass(frozen=True)
class BuildResult:
    projections: list[PlayerProjection]
    status: dict


@dataclass
class _History:
    # (stat row, efficiency weight, usage weight)
    rows: list[tuple[dict, float, float]] = field(default_factory=list)
    games: set[tuple[int, int]] = field(default_factory=set)
    last_team: str | None = None
    last_order: int = -1


def _value(row: dict, field: str) -> float:
    return float(number(row.get(field)) or 0.0)


def _position(value: str) -> str:
    pos = str(value or "").strip().upper()
    return "K" if pos in {"K", "PK"} else pos


def _target_group(value: str) -> str | None:
    pos = str(value or "").strip().upper()
    if pos == "FB":
        return "RB"
    return pos if pos in GROUP_TARGET_SHARE else None


def _name_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _order(season: int, week: int) -> int:
    # 24 leaves room for the postseason without making consecutive seasons
    # appear adjacent when a player's last game was in January.
    return season * 24 + week


def _prior(values: tuple[float, ...], rank: int) -> float:
    return values[min(max(rank - 1, 0), len(values) - 1)]


def _bayes(numerator: float, denominator: float, prior: tuple[float, float]) -> float:
    mean, pseudo = prior
    return (numerator + mean * pseudo) / (denominator + pseudo)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _round_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(max(0.0, value), 2) for key, value in metrics.items()}


def _history_index(
    rows: list[dict], *, season: int, week: int
) -> tuple[dict[str, _History], dict[tuple[int, int, str], dict[str, float]]]:
    target = _order(season, week)
    history: dict[str, _History] = {}
    team_games: dict[tuple[int, int, str], dict[str, float]] = defaultdict(
        lambda: {
            "attempts": 0.0,
            "pass_yards": 0.0,
            "carries": 0.0,
            "rush_yards": 0.0,
            "targets": 0.0,
            "receive_yards": 0.0,
            "targets_WR": 0.0,
            "targets_TE": 0.0,
            "targets_RB": 0.0,
        }
    )
    for row in rows:
        row_season = int(number(row.get("season")) or 0)
        row_week = int(number(row.get("week")) or 0)
        if not row_season or not row_week or _order(row_season, row_week) >= target:
            continue
        if str(row.get("season_type") or "REG").upper() != "REG":
            continue
        team = teams.canonical(row.get("team") or "")
        if not team:
            continue
        game_key = (row_season, row_week, team)
        team_games[game_key]["attempts"] += _value(row, "attempts")
        team_games[game_key]["pass_yards"] += _value(row, "passing_yards")
        team_games[game_key]["carries"] += _value(row, "carries")
        team_games[game_key]["rush_yards"] += _value(row, "rushing_yards")
        team_games[game_key]["targets"] += _value(row, "targets")
        team_games[game_key]["receive_yards"] += _value(row, "receiving_yards")
        group = _target_group(row.get("position") or "")
        if group:
            team_games[game_key][f"targets_{group}"] += _value(row, "targets")

        player_id = str(row.get("player_id") or "").strip()
        if not player_id or _position(row.get("position") or "") not in POSITIONS:
            continue
        age = target - _order(row_season, row_week)
        item = history.setdefault(player_id, _History())
        item.rows.append((
            row, 0.5 ** (age / HALF_LIFE_WEEKS), 0.5 ** (age / USAGE_HALF_LIFE_WEEKS)
        ))
        item.games.add((row_season, row_week))
        row_order = _order(row_season, row_week)
        if row_order > item.last_order:
            item.last_order = row_order
            item.last_team = team
    return history, team_games


def _team_environment(
    *,
    history_rows: list[dict],
    team_games: dict[tuple[int, int, str], dict[str, float]],
    season: int,
    week: int,
    team: str,
    opponent: str,
    game,
    home: bool,
) -> dict[str, float | str | None]:
    target = _order(season, week)
    opponent_by_game: dict[tuple[int, int, str], str] = {}
    for row in history_rows:
        row_season = int(number(row.get("season")) or 0)
        row_week = int(number(row.get("week")) or 0)
        row_team = teams.canonical(row.get("team") or "")
        if row_team and _order(row_season, row_week) < target:
            opponent_by_game.setdefault(
                (row_season, row_week, row_team),
                teams.canonical(row.get("opponent_team") or ""),
            )

    league_weight = 0.0
    league = defaultdict(float)
    offense_weight = 0.0
    offense = defaultdict(float)
    defense_weight = 0.0
    defense = defaultdict(float)
    for key, totals in team_games.items():
        row_season, row_week, row_team = key
        age = target - _order(row_season, row_week)
        weight = 0.5 ** (age / HALF_LIFE_WEEKS)
        league_weight += weight
        for metric, value in totals.items():
            league[metric] += weight * value
        if row_team == team:
            offense_weight += weight
            for metric, value in totals.items():
                offense[metric] += weight * value
        if opponent_by_game.get(key) == opponent:
            defense_weight += weight
            for metric, value in totals.items():
                defense[metric] += weight * value

    fallback = {
        "attempts": 33.4,
        "pass_yards": 236.0,
        "carries": 26.1,
        "rush_yards": 113.0,
        "targets": 31.4,
        "receive_yards": 236.0,
    }
    league_mean = {
        metric: league[metric] / league_weight if league_weight else value
        for metric, value in fallback.items()
    }

    def shrunk(acc: dict, weight: float, metric: str, pseudo: float) -> float:
        return (acc[metric] + pseudo * league_mean[metric]) / (weight + pseudo)

    team_mean = {metric: shrunk(offense, offense_weight, metric, 6.0) for metric in fallback}
    # Efficiency allowed by a defence is mostly noise (split-half k ~ 35 games).
    allowed = {metric: shrunk(defense, defense_weight, metric, 20.0) for metric in fallback}
    target_rate = _clamp(
        (0.6 * team_mean["targets"] + 0.4 * allowed["targets"])
        / max(1.0, 0.6 * team_mean["attempts"] + 0.4 * allowed["attempts"]),
        0.84,
        0.98,
    )
    pass_delta = (
        allowed["pass_yards"] / max(allowed["attempts"], 1.0)
        - league["pass_yards"] / max(league["attempts"], 1.0)
    )
    rush_delta = (
        allowed["rush_yards"] / max(allowed["carries"], 1.0)
        - league["rush_yards"] / max(league["carries"], 1.0)
    )
    receive_delta = (
        allowed["receive_yards"] / max(allowed["targets"], 1.0)
        - league["receive_yards"] / max(league["targets"], 1.0)
    )

    book_total = getattr(game, "book_total", None)
    book_margin = getattr(game, "book_margin", None)
    if book_total is not None and book_margin is not None:
        team_margin = float(book_margin if home else -book_margin)
        game_total = float(book_total)
        implied = (game_total + team_margin) / 2.0
        source = "DraftKings game total and spread"
    else:
        team_margin = float(getattr(game, "model_margin", 0.0) or 0.0)
        if not home:
            team_margin = -team_margin
        game_total = getattr(game, "projected_total", None)
        game_total = LEAGUE_TOTAL if game_total is None else float(game_total)
        implied = (getattr(game, "projected_home_score", None) if home
                   else getattr(game, "projected_away_score", None))
        source = "independent team model"

    # Game script. Measured on 2022-2025, a favourite runs ~0.14 more plays per
    # point of spread at a lower dropback rate; the two cancel for passes, so
    # the spread moves carries and the total moves pass attempts.
    volume = {}
    for metric in ("attempts", "carries"):
        coef = VOLUME_MODEL[metric]
        base = league_mean[metric]
        volume[metric] = (
            base
            + coef["offense"] * (_team_prior(team_games, season, week, team, metric,
                                             "offense", base, opponent_by_game) - base)
            + coef["defense"] * (_team_prior(team_games, season, week, opponent, metric,
                                             "defense", base, opponent_by_game) - base)
            + coef["spread"] * team_margin
            + coef["total"] * (game_total - LEAGUE_TOTAL)
        )
    attempts = _clamp(volume["attempts"], 25.0, 44.0)
    carries = _clamp(volume["carries"], 19.0, 35.0)
    groups = {}
    for group, league_share in GROUP_TARGET_SHARE.items():
        observed = weight = 0.0
        for (row_season, row_week, row_team), totals in team_games.items():
            if row_team != team or totals["targets"] <= 0:
                continue
            w = 0.5 ** ((target - _order(row_season, row_week)) / USAGE_HALF_LIFE_WEEKS)
            observed += w * totals[f"targets_{group}"] / totals["targets"]
            weight += w
        groups[group] = (observed + GROUP_PSEUDO_GAMES * league_share) / (
            weight + GROUP_PSEUDO_GAMES)
    group_total = sum(groups.values()) or 1.0
    return {
        "attempts": attempts,
        "carries": carries,
        "group_target_share": {g: v / group_total for g, v in groups.items()},
        "targets": attempts * target_rate,
        "pass_yards_per_attempt_delta": _clamp(pass_delta, -1.2, 1.2),
        "rush_yards_per_carry_delta": _clamp(rush_delta, -0.8, 0.8),
        "receive_yards_per_target_delta": _clamp(receive_delta, -1.2, 1.2),
        "implied_points": float(implied) if implied is not None else None,
        "source": source,
    }


def _team_prior(
    team_games: dict[tuple[int, int, str], dict[str, float]],
    season: int,
    week: int,
    team: str,
    metric: str,
    side: str,
    league: float,
    opponent_by_game: dict[tuple[int, int, str], str],
) -> float:
    """A team's per-game volume (offense) or volume allowed (defense).

    This season's games, weighted with a TEAM_HALF_LIFE_WEEKS half-life, are
    shrunk toward a prior of last season's deviation from the league, of which
    only TEAM_PRIOR's carry-over share survives the offseason.
    """
    pseudo, carry = TEAM_PRIOR[(side, metric)]
    current = weight = 0.0
    last_total = last_games = 0.0
    for key, totals in team_games.items():
        row_season, row_week, row_team = key
        owner = row_team if side == "offense" else opponent_by_game.get(key)
        if owner != team:
            continue
        if row_season == season and row_week < week:
            w = 0.5 ** ((week - row_week) / TEAM_HALF_LIFE_WEEKS)
            current += w * totals[metric]
            weight += w
        elif row_season == season - 1:
            last_total += totals[metric]
            last_games += 1
    prior = league if not last_games else league + carry * (last_total / last_games - league)
    return (current + pseudo * prior) / (weight + pseudo)


def _continuity(item: _History | None, current_team: str) -> tuple[str, float, str]:
    if item is None or not item.games:
        return "no NFL history", 0.0, "current depth slot; no prior NFL usage carried forward"
    games = len(item.games)
    if item.last_team != current_team:
        return (
            "changed teams",
            0.18,
            f"current depth slot dominates; prior usage discounted after {item.last_team} transfer",
        )
    if games >= 12:
        return "same team", 0.72, "current depth slot blended with established same-team usage"
    if games >= 5:
        return "same team", 0.58, "current depth slot blended with limited same-team usage"
    return "same team", 0.40, "current depth slot dominates a small same-team sample"


def _weighted_sum(item: _History | None, field: str) -> float:
    return 0.0 if item is None else sum(_value(row, field) * weight for row, weight, _ in item.rows)


def _weighted_games(item: _History | None) -> float:
    return 0.0 if item is None else sum(weight for _, weight, _ in item.rows)


def _usage_games(item: _History | None, team: str) -> float:
    """Usage-weighted games played for ``team`` (the evidence behind a share)."""
    if item is None:
        return 0.0
    return sum(w for row, _, w in item.rows if teams.canonical(row.get("team") or "") == team)


def _league_passing(histories: dict[str, _History]) -> dict[str, tuple[float, float]]:
    """QB rate priors centred on the recency-weighted league, not a constant.

    The pseudo-counts are large (a passer's own rate needs ~190-540 attempts
    to earn half weight), so the centre matters: a fixed 6.95 yards per
    attempt sat 0.1-0.2 below every season since 2023 and cost a starter ~5
    yards a game.
    """
    sums = defaultdict(float)
    for item in histories.values():
        for row, weight, _ in item.rows:
            if _position(row.get("position") or "") != "QB":
                continue
            for stat in ("attempts", "completions", "passing_yards", "passing_tds",
                          "passing_interceptions"):
                sums[stat] += weight * _value(row, stat)
    attempts = sums["attempts"]
    if attempts < 500:
        return {key: RATE_PRIORS[key] for key in
                ("completion", "pass_yards", "pass_td", "interception")}
    return {
        "completion": (sums["completions"] / attempts, RATE_PRIORS["completion"][1]),
        "pass_yards": (sums["passing_yards"] / attempts, RATE_PRIORS["pass_yards"][1]),
        "pass_td": (sums["passing_tds"] / attempts, RATE_PRIORS["pass_td"][1]),
        "interception": (sums["passing_interceptions"] / attempts,
                         RATE_PRIORS["interception"][1]),
    }


def _receiver_rates(item: _History | None, position: str) -> tuple[float, float]:
    """(catch rate, yards per target), each shrunk toward a prior read off the
    player's own shrunk aDOT rather than toward a flat positional mean."""
    league_adot, ypt_a, ypt_b, catch_a, catch_b = ADOT_PRIOR[position]
    targets = _weighted_sum(item, "targets")
    adot = (_weighted_sum(item, "receiving_air_yards") + ADOT_PSEUDO_TARGETS * league_adot) / (
        targets + ADOT_PSEUDO_TARGETS)
    key = position.lower()
    catch = _bayes(_weighted_sum(item, "receptions"), targets,
                   (_clamp(catch_a + catch_b * adot, 0.45, 0.90), RATE_PRIORS[f"catch_{key}"][1]))
    ypt = _bayes(_weighted_sum(item, "receiving_yards"), targets,
                 (_clamp(ypt_a + ypt_b * adot, 4.0, 12.0), RATE_PRIORS[f"receive_yards_{key}"][1]))
    return catch, ypt


def _snap_trends(
    snap_rows: list[dict], roster: list[dict], *, season: int, week: int
) -> dict[tuple[str, str], float]:
    """(team, gsis id) -> last same-team game snap % over its usage-weighted mean.

    A player whose snaps just rose (role change, teammate hurt) is about to see
    more volume than his trailing share says; one whose snaps fell, less.
    """
    pfr_to_gsis = {
        str(row.get("pfr_id") or "").strip(): str(row.get("gsis_id") or "").strip()
        for row in roster if row.get("pfr_id") and row.get("gsis_id")
    }
    target = _order(season, week)
    games: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in snap_rows:
        if str(row.get("game_type") or "REG").upper() != "REG":
            continue
        gsis = pfr_to_gsis.get(str(row.get("pfr_player_id") or "").strip())
        order = _order(int(number(row.get("season")) or 0), int(number(row.get("week")) or 0))
        pct = number(row.get("offense_pct"))
        if not gsis or pct is None or order >= target:
            continue
        games[(teams.canonical(row.get("team") or ""), gsis)].append((order, float(pct)))
    trends = {}
    for key, rows in games.items():
        rows.sort()
        weights = [0.5 ** ((target - order) / USAGE_HALF_LIFE_WEEKS) for order, _ in rows]
        mean = sum(w * pct for w, (_, pct) in zip(weights, rows)) / sum(weights)
        if mean > 0.05:
            trends[key] = _clamp(max(rows[-1][1], 0.02) / mean, 0.25, 4.0)
    return trends


def _share(
    item: _History | None,
    field: str,
    denominator: str,
    team_games: dict[tuple[int, int, str], dict[str, float]],
) -> float | None:
    if item is None:
        return None
    weighted = 0.0
    weight_sum = 0.0
    for row, _, weight in item.rows:
        key = (
            int(number(row.get("season")) or 0),
            int(number(row.get("week")) or 0),
            teams.canonical(row.get("team") or ""),
        )
        total = team_games.get(key, {}).get(denominator, 0.0)
        if total <= 0:
            continue
        weighted += weight * _value(row, field) / total
        weight_sum += weight
    return weighted / weight_sum if weight_sum else None


def _active_depth(
    roster: list[dict], depth: list[dict], injuries: list[dict]
) -> list[dict]:
    injury_by_id: dict[tuple[str, str], str] = {}
    injury_by_name: dict[tuple[str, str], str] = {}
    for row in injuries:
        team = teams.canonical(row.get("team") or "")
        player_id = str(row.get("gsis_id") or "").strip()
        name = _name_key(row.get("full_name") or row.get("player_name") or "")
        designation = str(row.get("report_status") or "").strip()
        if team and player_id:
            injury_by_id[(team, player_id)] = designation
        if team and name:
            injury_by_name[(team, name)] = designation
    active_by_id: dict[tuple[str, str], dict] = {}
    active_by_name: dict[tuple[str, str, str], dict] = {}
    for row in roster:
        team = teams.canonical(row.get("team") or "")
        position = _position(row.get("position") or row.get("depth_chart_position") or "")
        status = str(row.get("status") or "").upper()
        if team and position in POSITIONS and status == "ACT":
            player_id = str(row.get("gsis_id") or "").strip()
            if player_id:
                active_by_id[(team, player_id)] = row
            active_by_name[(team, position, _name_key(row.get("full_name") or ""))] = row

    selected: dict[tuple[str, str], dict] = {}
    for row in depth:
        team = teams.canonical(row.get("team") or "")
        position = _position(row.get("pos_abb") or row.get("pos_grp") or "")
        rank = int(number(row.get("pos_rank")) or 999)
        if position not in POSITIONS or rank > DEPTH_LIMITS[position]:
            continue
        player_id = str(row.get("gsis_id") or "").strip()
        roster_row = active_by_id.get((team, player_id)) if player_id else None
        if roster_row is None:
            roster_row = active_by_name.get(
                (team, position, _name_key(row.get("player_name") or ""))
            )
        if roster_row is None:
            continue
        resolved_id = str(roster_row.get("gsis_id") or player_id).strip()
        injury_status = injury_by_id.get((team, resolved_id)) or injury_by_name.get(
            (team, _name_key(roster_row.get("full_name") or row.get("player_name") or ""))
        )
        if str(injury_status or "").lower() == "out":
            continue
        key = (team, resolved_id or _name_key(row.get("player_name") or ""))
        candidate = {
            "team": team,
            "position": position,
            "depth_rank": rank,
            "depth_slot": str(row.get("pos_slot") or ""),
            "player_id": resolved_id,
            "player_name": str(
                roster_row.get("full_name") or row.get("player_name") or "Unknown"
            ),
            "status": str(roster_row.get("status") or "ACT"),
            "injury_status": injury_status or None,
            "headshot_url": str(roster_row.get("headshot_url") or ""),
        }
        previous = selected.get(key)
        if previous is None or rank < previous["depth_rank"]:
            selected[key] = candidate
    return sorted(
        selected.values(),
        key=lambda row: (row["team"], POSITIONS.index(row["position"]), row["depth_rank"]),
    )


def _confidence(item: _History | None, continuity: str) -> str:
    games = 0 if item is None else len(item.games)
    if continuity == "same team" and games >= 12:
        return "high"
    if games >= 5:
        return "medium"
    return "low"


def project(
    *,
    season: int,
    week: int,
    games: list[dict],
    game_projections: list,
    roster: list[dict],
    depth: list[dict],
    injuries: list[dict] | None = None,
    history_rows: list[dict],
    scheme_matchups: dict[tuple[str, str], object] | None = None,
    snap_rows: list[dict] | None = None,
) -> BuildResult:
    """Project QB/RB/WR/TE/K output for one slate."""
    snap_trend = _snap_trends(snap_rows or [], roster, season=season, week=week)
    histories, team_games = _history_index(history_rows, season=season, week=week)
    passing_priors = _league_passing(histories)
    current = _active_depth(roster, depth, injuries or [])
    active_teams = {teams.canonical(row.get("home_team") or "") for row in games}
    active_teams |= {teams.canonical(row.get("away_team") or "") for row in games}
    current = [row for row in current if row["team"] in active_teams]
    by_team: dict[str, list[dict]] = defaultdict(list)
    for player in current:
        by_team[player["team"]].append(player)

    projection_by_matchup = {(p.home, p.away): p for p in game_projections}
    rows_by_matchup = {
        (teams.canonical(row.get("home_team") or ""),
         teams.canonical(row.get("away_team") or "")): row
        for row in games
    }
    projections: list[PlayerProjection] = []
    for matchup, game_row in rows_by_matchup.items():
        home_team, away_team = matchup
        game = projection_by_matchup.get(matchup)
        if game is None:
            continue
        for team, opponent, home in (
            (away_team, home_team, False),
            (home_team, away_team, True),
        ):
            players = by_team.get(team, [])
            environment = _team_environment(
                history_rows=history_rows,
                team_games=team_games,
                season=season,
                week=week,
                team=team,
                opponent=opponent,
                game=game,
                home=home,
            )
            scheme_matchup = (scheme_matchups or {}).get((team, opponent))
            if scheme_matchup is not None:
                old_attempts = float(environment["attempts"])
                environment["attempts"] = _clamp(
                    old_attempts + float(scheme_matchup.pass_attempt_delta), 25.0, 44.0
                )
                target_rate = float(environment["targets"]) / max(old_attempts, 1.0)
                environment["targets"] = float(environment["attempts"]) * target_rate
                environment["carries"] = _clamp(
                    float(environment["carries"]) + float(scheme_matchup.carry_delta),
                    19.0,
                    35.0,
                )
                environment["pass_yards_per_attempt_delta"] = _clamp(
                    float(environment["pass_yards_per_attempt_delta"])
                    + float(scheme_matchup.pass_efficiency_delta),
                    -1.35,
                    1.35,
                )
                environment["receive_yards_per_target_delta"] = _clamp(
                    float(environment["receive_yards_per_target_delta"])
                    + float(scheme_matchup.pass_efficiency_delta),
                    -1.35,
                    1.35,
                )
                environment["rush_yards_per_carry_delta"] = _clamp(
                    float(environment["rush_yards_per_carry_delta"])
                    + float(scheme_matchup.rush_efficiency_delta),
                    -0.90,
                    0.90,
                )
                environment["source"] = str(environment["source"]) + " + scheme matrix"

            role_rows: list[dict] = []
            for player in players:
                item = histories.get(player["player_id"])
                continuity, persistence, reason = _continuity(item, team)
                player.update({
                    "history": item,
                    "continuity": continuity,
                    "persistence": persistence,
                    "reason": reason,
                })
                role_rows.append(player)

            # Target allocation is one constrained pool across RB/WR/TE.
            # Observed usage earns weight with same-team evidence; usage from
            # another team keeps the fixed transfer discount.
            for player in role_rows:
                if player["continuity"] == "same team":
                    evidence = _usage_games(player["history"], team)
                    player["usage_weight"] = evidence / (evidence + USAGE_PSEUDO_GAMES)
                else:
                    player["usage_weight"] = player["persistence"]

            def claim(player: dict, priors: dict, field: str) -> float:
                prior = _prior(priors[player["position"]], player["depth_rank"])
                observed = _share(player["history"], field, field, team_games)
                trend = snap_trend.get((team, player["player_id"]))
                if observed is not None and trend is not None:
                    observed *= trend ** SNAP_TREND_BETA[field]
                weight = player["usage_weight"]
                score = prior if observed is None else weight * observed + (1 - weight) * prior
                # A slot that often records nothing gives up its claim to
                # teammates rather than to nobody (see ACTIVE_RATE).
                return score * ACTIVE_RATE.get(
                    (player["position"], int(player["depth_rank"])), 1.0)

            # Targets: team -> pool (WR+TE, RB) -> player, so an absent back's
            # targets stay with the backfield (reports/PROPS_MODEL.md, B3).
            target_counts: dict[str, float] = {}
            group_share = dict(environment["group_target_share"])
            if scheme_matchup is not None:
                # The scheme matrix's position-level response sizes the pools
                # here and splits WR from TE inside the shared pool below.
                for group in group_share:
                    group_share[group] *= float(
                        scheme_matchup.target_multipliers.get(group, 1.0))
                scale = sum(group_share.values()) or 1.0
                group_share = {g: v / scale for g, v in group_share.items()}
            pools = []
            for positions in TARGET_GROUPS:
                share = sum(group_share[pos] for pos in positions)
                reserve = sum(group_share[pos] * GROUP_RESERVE[pos] for pos in positions)
                pools.append((
                    [row for row in role_rows if row["position"] in positions],
                    float(environment["targets"]) * (share - reserve),
                    scheme_matchup.target_multipliers if scheme_matchup else {},
                ))
            for members, pool, multipliers in pools:
                scores = {
                    player["player_id"]: claim(player, TARGET_PRIORS, "targets")
                    * float(multipliers.get(player["position"], 1.0))
                    for player in members
                }
                total_score = sum(scores.values())
                for player_id, score in scores.items():
                    target_counts[player_id] = pool * score / max(total_score, 0.01)

            backs = [row for row in role_rows if row["position"] == "RB"]
            carry_scores: dict[str, float] = {
                player["player_id"]: claim(player, CARRY_PRIORS, "carries")
                for player in backs
            }
            quarterback = next(
                (row for row in role_rows if row["position"] == "QB"), None
            )
            quarterback_item = None if quarterback is None else quarterback["history"]
            quarterback_games = _weighted_games(quarterback_item)
            quarterback_carries = (
                (_weighted_sum(quarterback_item, "carries") + 4.4 * 4.0)
                / (quarterback_games + 4.0)
            )
            # Reserve the starting QB's own forecast plus a small gadget/backup
            # pool before assigning RB work.  A fixed RB percentage overbooks a
            # Lamar Jackson backfield even if it looks harmless for pocket QBs.
            team_carries = float(environment["carries"])
            backfield_pool = max(
                team_carries * 0.55,
                team_carries - quarterback_carries - team_carries * CARRY_RESERVE,
            )
            carry_scale = backfield_pool / max(sum(carry_scores.values()), 0.01)

            for player in role_rows:
                item = player["history"]
                position = player["position"]
                weighted_games = _weighted_games(item)
                metrics: dict[str, float] = {}
                if position == "QB":
                    attempts = float(environment["attempts"]) * QB_ATTEMPT_SHARE
                    hist_attempts = _weighted_sum(item, "attempts")
                    completions = _bayes(
                        _weighted_sum(item, "completions"), hist_attempts,
                        passing_priors["completion"],
                    ) * attempts
                    passing_yards = _bayes(
                        _weighted_sum(item, "passing_yards"), hist_attempts,
                        passing_priors["pass_yards"],
                    )
                    passing_yards += 0.45 * float(
                        environment["pass_yards_per_attempt_delta"]
                    )
                    passing_yards *= attempts
                    passing_tds = _bayes(
                        _weighted_sum(item, "passing_tds"), hist_attempts,
                        passing_priors["pass_td"],
                    ) * attempts
                    passing_tds *= _clamp(
                        float(environment["implied_points"] or 22.5) / 22.5,
                        0.72,
                        1.35,
                    )
                    interceptions = _bayes(
                        _weighted_sum(item, "passing_interceptions"), hist_attempts,
                        passing_priors["interception"],
                    ) * attempts
                    rush_attempts = quarterback_carries
                    rush_yards = _bayes(
                        _weighted_sum(item, "rushing_yards"),
                        _weighted_sum(item, "carries"),
                        RATE_PRIORS["rush_yards_qb"],
                    )
                    rush_yards += 0.45 * float(
                        environment["rush_yards_per_carry_delta"]
                    )
                    rush_yards *= rush_attempts
                    metrics = {
                        "pass_attempts": attempts,
                        "completions": completions,
                        "passing_yards": passing_yards,
                        "passing_tds": passing_tds,
                        "interceptions": interceptions,
                        "rush_attempts": rush_attempts,
                        "rushing_yards": rush_yards,
                    }
                elif position in {"RB", "WR", "TE"}:
                    targets = target_counts[player["player_id"]]
                    catch, ypt = _receiver_rates(item, position)
                    ypt += 0.40 * float(environment["receive_yards_per_target_delta"])
                    metrics = {
                        "targets": targets,
                        "receptions": targets * catch,
                        "receiving_yards": targets * ypt,
                    }
                    if position == "RB":
                        carries = carry_scores[player["player_id"]] * carry_scale
                        ypc = _bayes(
                            _weighted_sum(item, "rushing_yards"),
                            _weighted_sum(item, "carries"),
                            RATE_PRIORS["rush_yards_rb"],
                        )
                        ypc += 0.45 * float(environment["rush_yards_per_carry_delta"])
                        metrics.update({"carries": carries, "rushing_yards": carries * ypc})

                    implied = float(environment["implied_points"] or 22.5)
                    rush_td_rate = 0.0
                    if position == "RB":
                        rush_td_rate = _bayes(
                            _weighted_sum(item, "rushing_tds"),
                            _weighted_sum(item, "carries"),
                            RATE_PRIORS["rush_td_rb"],
                        )
                    receive_td_rate = _bayes(
                        _weighted_sum(item, "receiving_tds"),
                        _weighted_sum(item, "targets"),
                        RATE_PRIORS[f"receive_td_{position.lower()}"],
                    )
                    touchdown_lambda = (
                        metrics.get("carries", 0.0) * rush_td_rate
                        + targets * receive_td_rate
                    ) * _clamp(implied / 22.5, 0.65, 1.45)
                    metrics["anytime_td_probability"] = _clamp(
                        1.0 - math.exp(-touchdown_lambda), 0.01, 0.82
                    )
                else:  # K
                    implied = float(environment["implied_points"] or 22.5)
                    observed_fg_attempts = (
                        _weighted_sum(item, "fg_att") + 1.85 * 6.0
                    ) / (weighted_games + 6.0)
                    fg_attempts_per_game = (
                        player["persistence"] * observed_fg_attempts
                        + (1 - player["persistence"]) * 1.85
                    )
                    fg_attempts = fg_attempts_per_game * _clamp(implied / 22.5, 0.70, 1.35)
                    fg_pct = _bayes(
                        _weighted_sum(item, "fg_made"),
                        _weighted_sum(item, "fg_att"),
                        (0.845, 24.0),
                    )
                    observed_pats = (
                        _weighted_sum(item, "pat_made") + 2.25 * 6.0
                    ) / (weighted_games + 6.0)
                    pats_per_game = (
                        player["persistence"] * observed_pats
                        + (1 - player["persistence"]) * 2.25
                    )
                    pat_made = pats_per_game * _clamp(implied / 22.5, 0.65, 1.40)
                    fg_made = fg_attempts * fg_pct
                    metrics = {
                        "fg_attempts": fg_attempts,
                        "fg_made": fg_made,
                        "pat_made": pat_made,
                        "kicking_points": 3.0 * fg_made + pat_made,
                    }

                game_id = str(game_row.get("game_id") or f"{season}_{week}_{away_team}_{home_team}")
                scheme_context: dict[str, object] = {}
                if scheme_matchup is not None:
                    scheme_context = {
                        "model_version": scheme_matchup.model_version,
                        "source_seasons": list(scheme_matchup.source_seasons),
                        "confidence": scheme_matchup.confidence,
                        "expected_man_rate": scheme_matchup.expected_man_rate,
                        "expected_zone_rate": scheme_matchup.expected_zone_rate,
                        "expected_motion_rate": scheme_matchup.expected_motion_rate,
                        "expected_play_action_rate": (
                            scheme_matchup.expected_play_action_rate
                        ),
                        "expected_blitz_rate": scheme_matchup.expected_blitz_rate,
                        "expected_pressure_rate": scheme_matchup.expected_pressure_rate,
                        "target_multiplier": scheme_matchup.target_multipliers.get(
                            position, 1.0
                        ),
                        "pass_attempt_delta": scheme_matchup.pass_attempt_delta,
                        "carry_delta": scheme_matchup.carry_delta,
                        "pass_efficiency_delta": scheme_matchup.pass_efficiency_delta,
                        "rush_efficiency_delta": scheme_matchup.rush_efficiency_delta,
                        "factors": list(scheme_matchup.factors),
                    }
                projections.append(PlayerProjection(
                    season=season,
                    week=week,
                    game_id=game_id,
                    kickoff=getattr(game, "kickoff", ""),
                    kickoff_utc=getattr(game, "kickoff_utc", ""),
                    team=team,
                    opponent=opponent,
                    home=home,
                    player_id=player["player_id"],
                    player_name=player["player_name"],
                    position=position,
                    depth_rank=player["depth_rank"],
                    depth_slot=player["depth_slot"],
                    roster_status=player["status"],
                    injury_status=player["injury_status"],
                    headshot_url=player["headshot_url"],
                    history_games=0 if item is None else len(item.games),
                    last_team=None if item is None else item.last_team,
                    role_continuity=player["continuity"],
                    persistence_weight=player["persistence"],
                    role_reason=player["reason"],
                    confidence=(
                        "low"
                        if str(player["injury_status"] or "").lower()
                        in {"questionable", "doubtful"}
                        else _confidence(item, player["continuity"])
                    ),
                    team_environment_source=str(environment["source"]),
                    implied_team_points=(None if environment["implied_points"] is None
                                         else round(float(environment["implied_points"]), 2)),
                    scheme_context=scheme_context,
                    metrics=_round_metrics(metrics),
                ))

    projections.sort(
        key=lambda row: (
            row.kickoff_utc,
            row.game_id,
            1 if row.home else 0,
            POSITIONS.index(row.position),
            row.depth_rank,
            row.player_name,
        )
    )
    latest_depth = max((str(row.get("dt") or "") for row in depth), default="")
    roster_weeks = [int(number(row.get("week")) or 0) for row in roster]
    status = {
        "model_version": MODEL_VERSION,
        "positions": list(POSITIONS),
        "players_projected": len(projections),
        "teams_covered": len({row.team for row in projections}),
        "active_roster_week": max(roster_weeks, default=0),
        "depth_chart_as_of": latest_depth or None,
        "history_rows": len(history_rows),
        "history_seasons": sorted({
            int(number(row.get("season")) or 0) for row in history_rows
            if number(row.get("season"))
        }),
        "injury_report": (
            f"{len(injuries or [])} current-week designations ingested"
            if injuries
            else "not yet published for the 2026 regular season"
        ),
        "role_policy": (
            "active roster and current depth determine next-game role; historical usage "
            "is discounted after team changes and excluded for players with no NFL history"
        ),
        "pricing": "none - projections only; no player-prop lines or edges",
        "scheme_matchups_applied": len({
            (row.team, row.opponent) for row in projections if row.scheme_context
        }),
    }
    return BuildResult(projections=projections, status=status)
