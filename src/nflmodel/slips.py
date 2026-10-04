"""Weekly prop slips: the line-shopping strategy as code.

The prop pricing (`prop_pricing`) gives every quoted line a calibrated lean.
This module turns those leans into a small set of pick'em slips (PrizePicks
style) under the rules the 2026 audit earned (reports/PROPS_MODEL.md, H):

1. **Markets with a record.** Only (position, market, side) combinations whose
   point-in-time replay record cleared 56% on at least 15 plays. RB catch and
   TE catch/yardage unders lead; RB carries unders (24-26) never qualify,
   because books price a bell-cow's workload better than a usage regression.
2. **Starters with history.** No depth 3+ skill players, no Questionable or
   Doubtful players, at least MIN_SEASON_GAMES games this season.
3. **No role expansion.** No under on a player whose role just grew: a recent
   regular at his position group is missing from this week's projections
   (Out, IR or inactive), or his team is starting a different quarterback.
4. **No line moved against the pick** by a full unit (or 10% of the line), no
   0.5 lines, and no under where the player cleared the line in two of his last
   three games (recent form; a judgment guard the ledger is measuring).
5. **Slips:** one leg per player, legs in a slip from different games, then
   the format that maximises the chance the combined payout clears the target
   (a Monte Carlo with a shared game shock and a shared week shock).

Payout multipliers are PrizePicks' standard ones as of October 2026 and vary
by state and promotion: verify in the app before entering.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from statistics import NormalDist

from . import prop_pricing
from .sources.nflverse import number

# side, position, market -> (wins, plays): 2026 weeks 1-3, matrix rebuilt
# point-in-time, starters, plays priced >= 53% (research/props_model/line_test.py).
MARKET_RECORD = {
    ("under", "RB", "receptions"): (33, 49),
    ("under", "RB", "receiving_yards"): (18, 26),
    ("under", "TE", "receiving_yards"): (25, 39),
    ("under", "TE", "receptions"): (36, 61),
    ("under", "QB", "passing_tds"): (19, 32),
    ("under", "QB", "rush_attempts"): (14, 25),
    ("under", "QB", "rushing_yards"): (13, 23),
    ("under", "QB", "completions"): (20, 39),
    ("under", "RB", "carries"): (24, 50),
    ("under", "RB", "rushing_yards"): (16, 32),
    ("under", "WR", "receptions"): (43, 81),
    ("under", "WR", "receiving_yards"): (27, 52),
    ("over", "QB", "passing_tds"): (10, 15),
    ("over", "RB", "carries"): (11, 21),
    ("over", "RB", "receptions"): (12, 21),
    ("over", "RB", "rushing_yards"): (17, 35),
    ("over", "TE", "receptions"): (8, 16),
    ("over", "WR", "receiving_yards"): (10, 16),
    ("over", "WR", "receptions"): (4, 10),
}
MIN_MARKET_HIT = 0.56
MIN_MARKET_PLAYS = 15
MIN_LEG_PROBABILITY = 0.53
MIN_SEASON_GAMES = 2
# 0.5 lines (anytime TD pass, one catch) carry juice ESPN does not show and are
# goblin/demon variants on pick'em sites, not standard lines.
MIN_LINE = 1.0
REGULAR_TARGET_SHARE = 0.15          # a missing regular of any receiving position
SAME_POSITION_TARGET_SHARE = 0.08    # a missing contributor at the leg's own position
REGULAR_CARRY_SHARE = 0.30
RECENT_GAMES = 3
# Recent form: an under is dropped when the player cleared the line in at least
# two of his last three games (an over, when he fell short). A judgment guard
# from the week-4 audit, not yet measured; the ledger grades the legs it drops.
FORM_GAMES = 3
FORM_AGAINST = 2
STAT_FIELD = {
    "receptions": "receptions", "receiving_yards": "receiving_yards", "targets": "targets",
    "carries": "carries", "rush_attempts": "carries", "rushing_yards": "rushing_yards",
    "passing_tds": "passing_tds", "passing_yards": "passing_yards",
    "completions": "completions", "pass_attempts": "attempts",
    "interceptions": "passing_interceptions",
}

# PrizePicks multipliers (verify in app): hits -> payout multiple of the stake.
PAYOUTS = {
    "power2": {2: 3.0}, "power3": {3: 6.0}, "power4": {4: 10.0},
    "flex3": {3: 3.0, 2: 1.0}, "flex4": {4: 6.0, 3: 1.5},
    "flex5": {5: 10.0, 4: 2.0, 3: 0.4}, "flex6": {6: 25.0, 5: 2.0, 4: 0.4},
}
FORMAT_SIZE = {name: int(name[-1]) for name in PAYOUTS}
# Owner's standing plan: six $25 entries in promo credits, cash payout to beat $50.
PLAN = {"entries": 6, "stake": 25.0, "target": 50.0,
        "formats": ("power2", "power3", "flex3", "flex4")}
# How much of a leg's calibrated edge the simulation trusts: the replay's
# leans priced >= 53% hit 53.7% against an average price of ~57%.
EDGE_TRUST = 0.53
SIMULATIONS = 20000
GAME_RHO = 0.15
WEEK_RHO = 0.05


@dataclass
class Leg:
    season: int
    week: int
    home: str
    away: str
    player: str
    player_id: str
    team: str
    opponent: str
    position: str
    game_id: str
    kickoff: str
    metric: str
    side: str            # over | under
    line: float
    projection: float
    probability: float
    market_record: tuple[int, int]

    @property
    def label(self) -> str:
        word = "More" if self.side == "over" else "Less"
        return f"{self.player} {word} {self.line:g} {self.metric.replace('_', ' ')}"


@dataclass
class SlipPlan:
    season: int
    week: int
    format: str | None
    entries: int
    stake: float
    target: float
    slips: list[list[Leg]] = field(default_factory=list)
    estimates: dict = field(default_factory=dict)
    alternatives: dict = field(default_factory=dict)
    excluded: dict = field(default_factory=dict)
    # Legs the rules dropped (lean >= 53% in a market with a record), kept so
    # the ledger can grade whether each rule earns its place.
    dropped: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        out = asdict(self)
        out["slips"] = [{"legs": [{**asdict(leg), "label": leg.label} for leg in slip],
                         "stake": self.stake, "format": self.format} for slip in self.slips]
        out["note"] = ("PrizePicks standard payouts assumed; verify the format's multiplier "
                       "and each line in the app. Estimates trust the model's edge at the "
                       "replay's measured rate; three weeks of evidence.")
        return out


def market_allowed(side: str, position: str, metric: str) -> bool:
    wins, plays = MARKET_RECORD.get((side, position, metric), (0, 0))
    return plays >= MIN_MARKET_PLAYS and wins / plays >= MIN_MARKET_HIT


def _season_games(player_results: list[dict], season: int, week: int) -> dict[str, int]:
    games: dict[str, set] = defaultdict(set)
    for row in player_results:
        row_week = int(number(row.get("week")) or 0)
        if int(number(row.get("season")) or 0) == season and row_week < week:
            games[str(row.get("player_id") or "")].add(row_week)
    return {pid: len(weeks) for pid, weeks in games.items()}


def role_changes(player_results: list[dict], player_projections: list, *, season: int,
                 week: int) -> dict[str, set[str]]:
    """team -> flags: 'receiver' (a regular target earner is missing this week),
    'receiver:<POS>' (a contributor at that position is missing), 'rusher' (a
    regular ball carrier is missing), 'quarterback' (a different starter from
    the team's recent top passer)."""
    by_team_week: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in player_results:
        row_week = int(number(row.get("week")) or 0)
        if int(number(row.get("season")) or 0) == season and row_week < week:
            by_team_week[str(row.get("team") or "")][row_week].append(row)
    projected = defaultdict(set)
    starters = {}
    for p in player_projections:
        projected[p.team].add(p.player_id)
        if p.position == "QB" and p.depth_rank == 1:
            starters[p.team] = p.player_id
    flags: dict[str, set[str]] = defaultdict(set)
    for team, weeks in by_team_week.items():
        recent = [weeks[w] for w in sorted(weeks)[-RECENT_GAMES:]]
        target_share: dict[str, list[float]] = defaultdict(list)
        carry_share: dict[str, list[float]] = defaultdict(list)
        passing: dict[str, float] = defaultdict(float)
        positions: dict[str, str] = {}
        for rows in recent:
            targets = sum(float(number(r.get("targets")) or 0) for r in rows) or 1.0
            carries = sum(float(number(r.get("carries")) or 0) for r in rows) or 1.0
            for r in rows:
                pid = str(r.get("player_id") or "")
                positions[pid] = "RB" if str(r.get("position") or "") == "FB" else str(
                    r.get("position") or "")
                target_share[pid].append(float(number(r.get("targets")) or 0) / targets)
                if str(r.get("position") or "") in ("RB", "FB"):
                    carry_share[pid].append(float(number(r.get("carries")) or 0) / carries)
                passing[pid] += float(number(r.get("attempts")) or 0)
        games = len(recent)
        for pid, shares in target_share.items():
            if pid in projected[team]:
                continue
            share = sum(shares) / games
            if share >= REGULAR_TARGET_SHARE:
                flags[team].add("receiver")
            if share >= SAME_POSITION_TARGET_SHARE and positions.get(pid):
                flags[team].add(f"receiver:{positions[pid]}")
        for pid, shares in carry_share.items():
            if sum(shares) / games >= REGULAR_CARRY_SHARE and pid not in projected[team]:
                flags[team].add("rusher")
        if passing and team in starters:
            usual = max(passing, key=passing.get)
            if passing[usual] > 0 and usual != starters[team]:
                flags[team].add("quarterback")
    return flags


def _blocked_by_role(leg_side: str, position: str, metric: str, team_flags: set[str]) -> bool:
    # A quarterback who is not his team's recent starter, either side: his own
    # numbers come from a backup's sample (Mariota for Jayden Daniels, week 4).
    if position == "QB" and "quarterback" in team_flags:
        return True
    if leg_side != "under":
        return False
    receiving = metric in ("receptions", "receiving_yards", "targets")
    if receiving and position in ("WR", "TE", "RB") and (
            "receiver" in team_flags or f"receiver:{position}" in team_flags):
        return True
    if "rusher" in team_flags and position == "RB":
        return True
    # A quarterback missing a target earner (or his lead back) runs and
    # scrambles more: no QB rushing under then.
    if position == "QB" and metric in ("rush_attempts", "rushing_yards", "carries") and (
            "receiver" in team_flags or "rusher" in team_flags):
        return True
    return "quarterback" in team_flags and receiving and position in ("RB", "TE")


def recent_values(player_results: list[dict], *, season: int, week: int
                  ) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in player_results:
        row_week = int(number(row.get("week")) or 0)
        if int(number(row.get("season")) or 0) == season and row_week < week:
            out[str(row.get("player_id") or "")].append(row)
    for rows in out.values():
        rows.sort(key=lambda r: int(number(r.get("week")) or 0))
    return out


def _form_against(rows: list[dict], metric: str, side: str, line: float) -> bool:
    field = STAT_FIELD.get(metric)
    recent = rows[-FORM_GAMES:]
    if not field or len(recent) < FORM_AGAINST:
        return False
    values = [float(number(r.get(field)) or 0) for r in recent]
    against = sum((v > line) if side == "under" else (v < line) for v in values)
    return against >= min(FORM_AGAINST, len(values))


def _moved_against(side: str, line: float, opened: float | None) -> bool:
    if opened is None:
        return False
    step = max(1.0, 0.10 * abs(float(opened)))
    move = float(line) - float(opened)
    return move <= -step if side == "under" else move >= step


def _started(kickoff: str | None, now: datetime | None) -> bool:
    if now is None or not kickoff:
        return False
    try:
        return datetime.fromisoformat(str(kickoff).replace("Z", "+00:00")) <= now
    except ValueError:
        return False


def candidate_legs(slate, now: datetime | None = None) -> tuple[list[Leg], dict, list[dict]]:
    from .best_bets import PROP_MARKETS
    from .sources.oddsapi import normalise

    players = [p for p in slate.player_projections]
    by_id = {p.player_id: p for p in players}
    by_name = {normalise(p.player_name): p for p in players}
    season_games = _season_games(slate.player_results, slate.season, slate.week)
    history = recent_values(slate.player_results, season=slate.season, week=slate.week)
    flags = role_changes(slate.player_results, players, season=slate.season, week=slate.week)
    excluded: dict[str, int] = defaultdict(int)
    dropped: list[dict] = []
    best: dict[str, Leg] = {}
    for quote in slate.player_prop_quotes:
        spec = PROP_MARKETS.get(quote.market)
        player = (by_id.get(quote.player_id) if getattr(quote, "player_id", None)
                  else by_name.get(normalise(quote.player_name)))
        if spec is None or player is None or _started(player.kickoff_utc, now):
            continue
        metric = spec[0]
        mean = player.metrics.get(metric)
        if metric == "rush_attempts" and mean is None:
            metric, mean = "carries", player.metrics.get("carries")
        priced = prop_pricing.price(metric, mean, quote.line)
        if priced is None:
            continue
        side = "over" if priced["p_over"] >= 0.5 else "under"
        prob = priced["p_over"] if side == "over" else 1.0 - priced["p_over"]
        reason = None
        if float(quote.line) < MIN_LINE:
            reason = "0.5 line"
        elif prob < MIN_LEG_PROBABILITY:
            reason = "lean below 53%"
        elif not market_allowed(side, player.position, metric):
            reason = "market without a record"
        elif player.position in ("WR", "TE", "RB") and int(player.depth_rank) >= 3:
            reason = "backup (depth 3+)"
        elif str(player.injury_status or "").lower() in ("questionable", "doubtful", "out"):
            reason = "injury designation"
        elif season_games.get(player.player_id, 0) < MIN_SEASON_GAMES:
            reason = "under two games this season"
        elif _blocked_by_role(side, player.position, metric, flags.get(player.team, set())):
            reason = "role expansion (teammate or QB change)"
        elif _moved_against(side, quote.line, getattr(quote, "open_line", None)):
            reason = "line moved against the pick"
        elif _form_against(history.get(player.player_id, []), metric, side, float(quote.line)):
            reason = "recent form against the pick"
        home = player.team if player.home else player.opponent
        away = player.opponent if player.home else player.team
        record = MARKET_RECORD.get((side, player.position, metric), (0, 0))
        leg = Leg(slate.season, slate.week, home, away, player.player_name, player.player_id,
                  player.team, player.opponent, player.position, player.game_id,
                  player.kickoff_utc, metric, side, float(quote.line), round(float(mean), 2),
                  round(prob, 4), record)
        if reason:
            excluded[reason] += 1
            if reason not in ("lean below 53%", "market without a record", "0.5 line"):
                dropped.append({**asdict(leg), "label": leg.label, "reason": reason})
            continue
        held = best.get(player.player_id)
        if held is None or leg.probability > held.probability:
            best[player.player_id] = leg
    legs = sorted(best.values(), key=lambda leg: -leg.probability)
    return legs, dict(excluded), dropped


def build_slips(legs: list[Leg], size: int, entries: int) -> list[list[Leg]]:
    """Snake-draft the strongest legs into slips, one leg per game per slip.

    A leg that fits no open slip (its game is already in each) is skipped and
    the draft reaches further down the list, until every slip is full.
    """
    slips: list[list[Leg]] = [[] for _ in range(entries)]
    order = list(range(entries)) + list(range(entries))[::-1]
    turn = 0
    for leg in legs:
        if all(len(slip) == size for slip in slips):
            break
        for _ in range(2 * entries):
            slip = slips[order[turn % len(order)]]
            turn += 1
            if len(slip) < size and all(other.game_id != leg.game_id for other in slip):
                slip.append(leg)
                break
    return [slip for slip in slips if len(slip) == size]


def simulate(slips: list[list[Leg]], fmt: str, stake: float, target: float,
             runs: int = SIMULATIONS, seed: int = 20261004) -> dict:
    rng = random.Random(seed)
    legs = {(leg.player_id, leg.metric): leg for slip in slips for leg in slip}
    games = sorted({leg.game_id for leg in legs.values()})
    index = {game: i for i, game in enumerate(games)}
    normal = NormalDist()
    threshold = {key: normal.inv_cdf(0.5 + EDGE_TRUST * (leg.probability - 0.5))
                 for key, leg in legs.items()}
    own = math.sqrt(1 - GAME_RHO - WEEK_RHO)
    above = zero = 0
    total = 0.0
    for _ in range(runs):
        week = rng.gauss(0, 1)
        shocks = [rng.gauss(0, 1) for _ in games]
        won = {}
        for key, leg in legs.items():
            z = (math.sqrt(WEEK_RHO) * week + math.sqrt(GAME_RHO) * shocks[index[leg.game_id]]
                 + own * rng.gauss(0, 1))
            won[key] = (z if leg.side == "under" else -z) < threshold[key]
        payout = sum(
            stake * PAYOUTS[fmt].get(sum(won[(leg.player_id, leg.metric)] for leg in slip), 0.0)
            for slip in slips)
        total += payout
        above += payout > target
        zero += payout == 0
    return {"p_payout_above_target": round(above / runs, 3), "p_zero": round(zero / runs, 3),
            "expected_payout": round(total / runs, 2)}


def plan(slate, config: dict | None = None, now: datetime | None = None) -> SlipPlan:
    config = {**PLAN, **(config or {})}
    legs, excluded, dropped = candidate_legs(slate, now)
    result = SlipPlan(slate.season, slate.week, None, config["entries"], config["stake"],
                      config["target"], excluded=excluded, dropped=dropped)
    best = None
    for fmt in config["formats"]:
        slips = build_slips(legs, FORMAT_SIZE[fmt], config["entries"])
        if len(slips) < config["entries"]:
            result.alternatives[fmt] = {"unavailable": f"only {len(slips)} full slips"}
            continue
        estimate = simulate(slips, fmt, config["stake"], config["target"])
        result.alternatives[fmt] = estimate
        key = (estimate["p_payout_above_target"], estimate["expected_payout"])
        if best is None or key > best[0]:
            best = (key, fmt, slips, estimate)
    if best is not None:
        _, result.format, result.slips, result.estimates = best
    return result
