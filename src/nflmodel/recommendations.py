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


# League points per offensive yard (2021-2025, ~22.4 points on ~311 yards): turns
# the scheme matrix's yardage deltas into the same unit as a total.
POINTS_PER_YARD = 0.072


def _scheme_points(slate, team: str, opponent: str) -> float | None:
    """The scheme matrix's matchup deltas for `team`'s offense, in points."""
    m = slate.scheme_matchups.get((team, opponent))
    fields = ("pass_attempt_delta", "carry_delta", "pass_efficiency_delta",
              "rush_efficiency_delta")
    if m is None or any(getattr(m, f, None) is None for f in fields):
        return None
    yards = (float(m.pass_attempt_delta) * 6.6 + float(m.carry_delta) * 4.3
             + float(m.pass_efficiency_delta) * 33.0 + float(m.rush_efficiency_delta) * 26.0)
    return yards * POINTS_PER_YARD


def _season_only(slate, p) -> tuple[float | None, float | None]:
    """Total and home margin from this season's form alone, through the same matrix,
    shrink and quarterback adjustments as the model."""
    from . import matrix, totals

    forms = getattr(slate, "current_forms", None) or {}
    home, away = forms.get(p.home), forms.get(p.away)
    if home is None or away is None:
        return None, None
    home_pts = matrix.points(home, away, home=not p.neutral)
    away_pts = matrix.points(away, home, home=False)
    if home_pts is None or away_pts is None:
        return None, None
    total = totals.shrink_total(home_pts + away_pts) + float(p.availability_total or 0.0)
    return total, home_pts - away_pts + float(p.availability_margin or 0.0)


def _sign(value: float | None, tol: float = 0.5) -> int:
    if value is None or abs(value) < tol:
        return 0
    return 1 if value > 0 else -1


def reconcile(model_gap: float, signals: dict) -> list[str]:
    """Names of the signals that point the other way from the model's gap."""
    side = _sign(model_gap, 0.0)
    return [name for name, value in signals.items() if _sign(value) == -side]


def _live_note(slate, p) -> str:
    shares = [(getattr(slate, "live_share", None) or {}).get(t) for t in (p.away, p.home)]
    shares = [x for x in shares if x is not None]
    if not shares:
        return ""
    share = sum(shares) / len(shares)
    return f"scoring form weights 2026 at {share:.0%} and prior seasons at {1 - share:.0%}"


def _closing(conflicts: list[str]) -> str:
    if conflicts:
        return f" Points the other way: {', '.join(conflicts)}."
    return " No signal points the other way."


def _game_lists(slate) -> tuple[list[dict], list[dict]]:
    """Where the model's game numbers differ from the market, and why.

    These are gaps, not calls: the game model has shown no skill against the
    closing line (calibration.SKILL). Each row says what produces the model
    number, what this season's form alone says, what the matchup context
    implies in points, and names every signal that points the other way.
    """
    spreads, totals_rows = [], []
    for p in slate.projections:
        season_total, season_margin = _season_only(slate, p)
        live = _live_note(slate, p)
        away_ctx = _scheme_points(slate, p.away, p.home)
        home_ctx = _scheme_points(slate, p.home, p.away)
        if p.market_gap is not None and p.comparison_margin is not None:
            side = p.home if p.market_gap > 0 else p.away
            sign = 1 if side == p.home else -1
            market_line = -p.comparison_margin if side == p.home else p.comparison_margin
            model_line = -p.model_margin if side == p.home else p.model_margin
            context = None if home_ctx is None or away_ctx is None else home_ctx - away_ctx
            season_gap = None if season_margin is None else season_margin - p.comparison_margin
            conflicts = reconcile(p.market_gap, {"this season's form": season_gap,
                                                 "matchup context": context})
            drivers = [f"Model rates {side} {abs(p.market_gap):.1f} points better than the market"]
            if p.rating_margin is not None and p.efficiency_margin is not None:
                drivers[0] += (f" (home margin: power ratings {p.rating_margin:+.1f}, "
                               f"efficiency {p.efficiency_margin:+.1f}, blended 50/50)")
            if p.qb_out:
                drivers.append(f"QB out: {', '.join(p.qb_out)} ({p.availability_margin:+.1f})")
            if season_margin is not None:
                drivers.append(f"2026 form alone: home margin {season_margin:+.1f}")
            if context is not None:
                drivers.append(f"matchup context {sign * context:+.1f} points toward {side} "
                               "(scheme description, not a model input)")
            spreads.append({
                "game": f"{p.away} @ {p.home}",
                "selection": f"Model favours {side}",
                "market": f"{side} {market_line:+.1f}",
                "model": f"{side} {model_line:+.1f}",
                "season_only": (None if season_margin is None
                                else f"{side} {-sign * season_margin:+.1f}"),
                "gap": round(abs(p.market_gap), 2),
                "direction": "home" if p.market_gap > 0 else "away",
                "book": p.book_name or "nflverse consensus",
                "reason": "; ".join(drivers) + "." + _closing(conflicts),
                "conflicts": conflicts,
                "status": "model_gap",
            })
        if p.total_gap is not None:
            context = None if home_ctx is None or away_ctx is None else home_ctx + away_ctx
            season_gap = None if season_total is None else season_total - p.comparison_total
            conflicts = reconcile(p.total_gap, {"this season's form": season_gap,
                                                "matchup context": context})
            word = "higher" if p.total_gap > 0 else "lower"
            drivers = [f"Model {p.projected_total:.1f} ({p.away} {p.projected_away_score:.1f}, "
                       f"{p.home} {p.projected_home_score:.1f})" + (f": {live}" if live else "")]
            if p.qb_out:
                drivers.append(f"QB out: {', '.join(p.qb_out)} ({p.availability_total:+.1f})")
            if season_total is not None:
                drivers.append(f"2026 form alone: {season_total:.1f}")
            if context is not None:
                drivers.append(f"matchup context {context:+.1f} points (scheme description, "
                               "not a model input)")
            totals_rows.append({
                "game": f"{p.away} @ {p.home}",
                "selection": f"Model {word}",
                "market": round(p.comparison_total, 1),
                "model": round(p.projected_total, 1),
                "season_only": None if season_total is None else round(season_total, 1),
                "gap": round(abs(p.total_gap), 2),
                "book": p.book_name or "nflverse consensus",
                "reason": "; ".join(drivers) + "." + _closing(conflicts),
                "conflicts": conflicts,
                "status": "model_gap",
            })
    return (
        sorted(spreads, key=lambda row: -row["gap"]),
        sorted(totals_rows, key=lambda row: -row["gap"]),
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
    from . import prop_pricing, slips

    # The same price and the same evidence rules as best bets and slips, so the
    # list can never call a side the picks do not (it used to compare the raw
    # projection with the line, ignoring the outcome skew and the rules).
    allowed = {(leg.player_id, leg.metric) for leg in slips.candidate_legs(slate)[0]}
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
        priced = prop_pricing.price(metric, model, quote.line)
        if priced is None:
            continue
        over = priced["p_over"] >= 0.5
        leg_metric = "carries" if metric == "rush_attempts" and player.position == "RB" else metric
        playable = (player.player_id, leg_metric) in allowed
        quoted.append({
            "player": player.player_name,
            "team": player.team,
            "opponent": player.opponent,
            "position": player.position,
            "market_key": quote.market,
            "market": label,
            "selection": "OVER" if over else "UNDER",
            "probability": round(priced["p_over"] if over else 1 - priced["p_over"], 3),
            "playable": playable,
            "line": quote.line,
            "price": quote.over_price if over else quote.under_price,
            "model": round(float(model), 2),
            "gap": round(abs(gap), 2),
            "score": round(abs(priced["p_over"] - 0.5), 3),
            "scheme_score": round(_player_scheme_score(player), 3),
            "reason": _profile_reason(slate, player.team, player.opponent),
            "book": quote.book_title,
            "last_update": quote.last_update,
            "status": "quoted_model_gap",
        })
    quoted.sort(key=lambda row: (-row["playable"], -row["score"], -row["scheme_score"]))

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
    from . import weekly_props

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
        "weekly_position_props": weekly_props.for_slate(slate),
    }
