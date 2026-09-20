"""Ranked weekly watchlists with scheme-grounded explanations.

These rankings answer "where does the model disagree most?" They do not turn a
research-only model into a profitable betting system. Every row carries the
authority and distinguishes a quoted market gap from a projection-only standout.
"""

from __future__ import annotations

from .sources.oddsapi import normalise

PROP_METRICS = {
    "player_pass_attempts": ("pass_attempts", "Pass attempts", 3.0),
    "player_pass_yds": ("passing_yards", "Passing yards", 25.0),
    "player_pass_tds": ("passing_tds", "Passing TDs", 0.55),
    "player_rush_attempts": ("rush_attempts", "Rush attempts", 2.5),
    "player_rush_yds": ("rushing_yards", "Rushing yards", 11.0),
    "player_receptions": ("receptions", "Receptions", 1.1),
    "player_reception_yds": ("receiving_yards", "Receiving yards", 12.0),
}


def _handicap(margin: float, home: str, away: str) -> str:
    if abs(margin) < 0.05:
        return "PK"
    favourite = home if margin > 0 else away
    return f"{favourite} {-abs(margin):.1f}"


def _profile_reason(slate, team: str, opponent: str) -> str:
    profile = slate.scheme_profiles.get(team)
    defense = slate.scheme_profiles.get(opponent)
    matchup = slate.scheme_matchups.get((team, opponent))
    if not profile or not defense or not matchup:
        return "Scheme profile unavailable."
    formations = {
        "shotgun": profile.offense.get("formation_shotgun_rate", 0.0),
        "under center": profile.offense.get("formation_under_center_rate", 0.0),
        "pistol": profile.offense.get("formation_pistol_rate", 0.0),
    }
    personnel = {
        "11 personnel": profile.offense.get("personnel_11_rate", 0.0),
        "12 personnel": profile.offense.get("personnel_12_rate", 0.0),
        "21 personnel": profile.offense.get("personnel_21_rate", 0.0),
        "13 personnel": profile.offense.get("personnel_13_rate", 0.0),
    }
    formation = max(formations, key=formations.get)
    grouping = max(personnel, key=personnel.get)
    coverage = max(matchup.expected_coverages, key=matchup.expected_coverages.get)
    regime_flags = getattr(profile, "regime_flags", ())
    trend = regime_flags[0] if regime_flags else "no large live-regime flag"
    return (
        f"{team} uses {formation} {formations[formation]:.0%} and {grouping} "
        f"{personnel[grouping]:.0%}; {opponent} projects {matchup.expected_man_rate:.0%} man / "
        f"{matchup.expected_zone_rate:.0%} zone with {coverage.replace('_', ' ')} most common. "
        f"Blitz {matchup.expected_blitz_rate:.0%}, pressure {matchup.expected_pressure_rate:.0%}; "
        f"live trend: {trend}."
    )


def _game_lists(slate) -> tuple[list[dict], list[dict]]:
    spreads, totals = [], []
    for projection in slate.projections:
        if projection.market_gap is not None and projection.comparison_margin is not None:
            side = projection.home if projection.market_gap > 0 else projection.away
            market_line = (
                -projection.comparison_margin
                if side == projection.home else projection.comparison_margin
            )
            model_line = (
                -projection.model_margin if side == projection.home else projection.model_margin
            )
            spreads.append({
                "game": f"{projection.away} @ {projection.home}",
                "selection": side,
                "market": f"{side} {market_line:+.1f}",
                "model": f"{side} {model_line:+.1f}",
                "gap": round(abs(projection.market_gap), 2),
                "direction": "home" if projection.market_gap > 0 else "away",
                "book": projection.book_name or "nflverse consensus",
                "reason": _profile_reason(
                    slate, side, projection.away if side == projection.home else projection.home
                ),
                "status": "model_gap",
            })
        if projection.total_gap is not None:
            totals.append({
                "game": f"{projection.away} @ {projection.home}",
                "selection": "OVER" if projection.total_gap > 0 else "UNDER",
                "market": round(projection.comparison_total, 1),
                "model": round(projection.projected_total, 1),
                "gap": round(abs(projection.total_gap), 2),
                "book": projection.book_name or "nflverse consensus",
                "reason": (
                    _profile_reason(slate, projection.away, projection.home) + " "
                    + _profile_reason(slate, projection.home, projection.away)
                ),
                "status": "model_gap",
            })
    return (
        sorted(spreads, key=lambda row: -row["gap"]),
        sorted(totals, key=lambda row: -row["gap"]),
    )


def _player_scheme_score(player) -> float:
    context = player.scheme_context or {}
    target = float(context.get("target_multiplier") or 1.0) - 1.0
    pass_eff = float(context.get("pass_efficiency_delta") or 0.0)
    rush_eff = float(context.get("rush_efficiency_delta") or 0.0)
    pass_volume = float(context.get("pass_attempt_delta") or 0.0) / 20.0
    carry_volume = float(context.get("carry_delta") or 0.0) / 16.0
    if player.position == "QB":
        return pass_eff + pass_volume
    if player.position == "RB":
        return 0.55 * rush_eff + 0.25 * carry_volume + 0.20 * target
    return 0.70 * pass_eff + 0.30 * target


def _primary_markets(player) -> tuple[str, ...]:
    return {
        "QB": ("player_pass_yds",),
        "RB": ("player_rush_yds",),
        "WR": ("player_reception_yds",),
        "TE": ("player_reception_yds",),
    }.get(player.position, ())


def _metric_value(player, market: str, metric: str):
    if market == "player_rush_attempts" and player.position == "RB":
        metric = "carries"
    return player.metrics.get(metric)


def _player_lists(slate) -> tuple[list[dict], list[dict]]:
    projections = {
        normalise(player.player_name): player
        for player in slate.player_projections
        if player.position in {"QB", "RB", "WR", "TE"}
        and str(player.injury_status or "").lower() != "out"
    }
    quoted = []
    for quote in slate.player_prop_quotes:
        player = projections.get(normalise(quote.player_name))
        spec = PROP_METRICS.get(quote.market)
        if player is None or spec is None:
            continue
        metric, label, scale = spec
        model = _metric_value(player, quote.market, metric)
        if model is None:
            continue
        gap = float(model) - quote.line
        quoted.append({
            "player": player.player_name,
            "team": player.team,
            "opponent": player.opponent,
            "position": player.position,
            "market_key": quote.market,
            "market": label,
            "selection": "OVER" if gap > 0 else "UNDER",
            "line": quote.line,
            "price": quote.over_price if gap > 0 else quote.under_price,
            "model": round(float(model), 2),
            "gap": round(abs(gap), 2),
            "score": round(abs(gap) / scale, 3),
            "scheme_score": round(_player_scheme_score(player), 3),
            "reason": _profile_reason(slate, player.team, player.opponent),
            "book": quote.book_title,
            "last_update": quote.last_update,
            "status": "quoted_model_gap",
        })
    quoted.sort(key=lambda row: (-row["score"], -row["scheme_score"]))

    standouts = []
    for player in projections.values():
        for market in _primary_markets(player):
            metric, label, _ = PROP_METRICS[market]
            model = _metric_value(player, market, metric)
            if model is None:
                continue
            standouts.append({
                "player": player.player_name,
                "team": player.team,
                "opponent": player.opponent,
                "position": player.position,
                "market_key": market,
                "market": label,
                "selection": None,
                "line": None,
                "price": None,
                "model": round(float(model), 2),
                "gap": None,
                "score": 0.0,
                "scheme_score": round(_player_scheme_score(player), 3),
                "reason": _profile_reason(slate, player.team, player.opponent),
                "book": None,
                "last_update": None,
                "status": "projection_only",
            })
    for row in standouts:
        peers = [
            candidate["model"] for candidate in standouts
            if candidate["market_key"] == row["market_key"]
        ]
        percentile = sum(value <= row["model"] for value in peers) / max(len(peers), 1)
        row["projection_percentile"] = round(percentile, 3)
        row["score"] = round(0.65 * row["scheme_score"] + 0.35 * percentile, 3)
    standouts.sort(key=lambda row: (-row["score"], -row["scheme_score"]))
    return quoted, standouts


def build_report(slate, *, limit: int = 10) -> dict:
    spreads, totals = _game_lists(slate)
    props, player_matchups = _player_lists(slate)
    team_matchups = []
    for (key_team, key_opponent), matchup in slate.scheme_matchups.items():
        team = getattr(matchup, "team", key_team)
        opponent = getattr(matchup, "opponent", key_opponent)
        score = (
            getattr(matchup, "pass_efficiency_delta", 0.0)
            + getattr(matchup, "rush_efficiency_delta", 0.0)
            + getattr(matchup, "pass_attempt_delta", 0.0) / 10.0
            + getattr(matchup, "carry_delta", 0.0) / 10.0
        )
        team_matchups.append({
            "team": team,
            "opponent": opponent,
            "score": round(score, 3),
            "pass_efficiency_delta": getattr(matchup, "pass_efficiency_delta", 0.0),
            "rush_efficiency_delta": getattr(matchup, "rush_efficiency_delta", 0.0),
            "pass_attempt_delta": getattr(matchup, "pass_attempt_delta", 0.0),
            "carry_delta": getattr(matchup, "carry_delta", 0.0),
            "reason": _profile_reason(slate, team, opponent),
            "confidence": matchup.confidence,
        })
    team_matchups.sort(key=lambda row: -row["score"])

    best = []
    for kind, rows in (("spread", spreads), ("total", totals), ("player_prop", props)):
        for row in rows[:3]:
            score = row.get("score") if kind == "player_prop" else row.get("gap", 0) / 3.0
            best.append({"type": kind, "rank_score": round(score, 3), **row})
    best.sort(key=lambda row: -row["rank_score"])
    return {
        "authority": slate.authority.level.value,
        "may_bet": slate.authority.may_bet,
        "label": "MODEL GAP WATCHLIST — NOT A VALIDATED EDGE",
        "method": (
            "Ranked by absolute model-versus-market gap; player gaps are scaled by "
            "market family. Scheme scores use bounded opportunity and efficiency deltas."
        ),
        "top_spreads": spreads[:limit],
        "top_totals": totals[:limit],
        "top_player_props": props[:limit],
        "projection_only_player_props": player_matchups[:limit],
        "best_team_matchups": team_matchups[:limit],
        "best_player_matchups": player_matchups[:limit],
        "best_bet_report": best[:limit],
    }
