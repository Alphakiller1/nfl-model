"""Player-specific football evidence behind a weekly threshold.

Observed tendencies, fitted model inputs, proxies and unobserved assignments
stay distinct. Splits contextualize the existing projection; they are never
multiplied into it again. A coverage or gap is a distribution, not a guarantee
of the opponent's next call or an individual defender assignment.
"""

from __future__ import annotations

import math

from . import prop_matchup_paths, teams
from .sources.oddsapi import normalise

MISSING_ASSIGNMENTS = (
    "route tree, slot/wide alignment and route participation by individual player",
    "press/bail, bracket/double-team, shadow assignment and coverage rotation",
    "zone/gap/power/counter blocking family and individual blocking assignments",
    "defensive front technique, stunt/twist and run-fit responsibility",
    "coach fourth-down decision model, kicker distance mix and live wind forecast",
)
GROUPS = {
    "intent": ("neutral_pass_rate", "no_huddle_rate"),
    "formation": ("formation_shotgun_rate", "formation_under_center_rate", "formation_pistol_rate"),
    "personnel": (
        "personnel_11_rate",
        "personnel_12_rate",
        "personnel_21_rate",
        "personnel_13_rate",
        "personnel_22_rate",
    ),
    "concepts": ("motion_rate", "play_action_rate", "rpo_rate", "screen_rate"),
    "coverage": (
        "man_rate",
        "zone_rate",
        "cover_0_rate",
        "cover_1_rate",
        "cover_2_rate",
        "cover_3_rate",
        "cover_4_rate",
        "cover_6_rate",
        "cover_2_man_rate",
        "single_high_rate",
        "two_high_rate",
    ),
    "front": (
        "blitz_rate",
        "pressure_rate",
        "stacked_box_rate",
        "light_box_rate",
        "avg_box",
        "base_rate",
        "nickel_rate",
        "dime_rate",
        "sub_package_rate",
    ),
    "response": (
        "pass_epa",
        "rush_epa",
        "pass_success_rate",
        "rush_success_rate",
        "pass_epa_man",
        "pass_epa_zone",
        "pass_epa_blitz",
        "pass_epa_no_blitz",
        "pass_epa_pressure",
        "pass_epa_play_action",
        "pass_epa_motion",
        "pass_epa_screen",
        "rush_epa_stacked_box",
        "rush_epa_light_box",
    ),
}


def context_game(slate, player) -> dict:
    for game in (getattr(slate, "prop_context", {}) or {}).get("games", []):
        home, away = teams.canonical(game.get("home", "")), teams.canonical(game.get("away", ""))
        if {home, away} == {player.team, player.opponent}:
            # The source can contain next week's fixture with the same clubs.
            from .sources.chase_context import timestamp

            left, right = timestamp(game.get("kickoff_utc")), timestamp(player.kickoff_utc)
            if left and right and abs((left - right).total_seconds()) < 6 * 3600:
                return game
    return {}


def _side(game: dict, team: str) -> str:
    return "home" if teams.canonical(game.get("home", "")) == team else "away"


def _club(game: dict, team: str, key: str):
    return game.get(f"{_side(game, team)}_{key}") or {}


def _phase(slate, game: dict, team: str, phase: str) -> tuple[dict, dict]:
    values, provenance = {}, {}
    # Current data wins field by field. An absent current shell retains its
    # prior-season date rather than inheriting the current source label.
    for window in ("scheme_current", "scheme"):
        profile = _club(game, team, window)
        if not isinstance(profile, dict):
            continue
        for group, rates in (profile.get(phase) or {}).items():
            if not isinstance(rates, dict):
                continue
            for key, value in rates.items():
                if isinstance(value, (float, int)) and not isinstance(value, bool):
                    if key not in values:
                        values[key] = value
                        provenance[key] = {
                            "source": "MLBMA observed snapshot",
                            "window": window,
                            "source_seasons": profile.get("source_seasons", []),
                            "participation_source_seasons": profile.get(
                                "participation_source_seasons", []
                            ),
                            "plays": profile.get(f"{phase}_plays"),
                            "group": group,
                        }
    local = (getattr(slate, "scheme_profiles", {}) or {}).get(team)
    if local:
        for key, value in getattr(local, phase, {}).items():
            if key not in values:
                values[key] = value
                provenance[key] = {
                    "source": "nfl-model pre-week scheme profile",
                    "source_seasons": list(local.source_seasons),
                    "participation_source_seasons": list(local.participation_source_seasons),
                    "charting_source_seasons": list(local.charting_source_seasons),
                    "plays": getattr(local, f"{phase}_plays"),
                    "coverage_samples": local.coverage_samples,
                }
    return values, provenance


def _record(
    family: str, values: dict, provenance: dict, keys: tuple[str, ...], mechanism: str
) -> dict:
    available = {key: values[key] for key in keys if key in values}
    return {
        "family": family,
        "status": "observed" if available else "unavailable",
        "values": available,
        "provenance": {key: provenance[key] for key in available},
        "missing_fields": [key for key in keys if key not in available],
        "mechanism": mechanism,
        "used_to_change_projection": False,
    }


def _splits(game: dict, player, key: str, family: str | None = None) -> dict:
    out, seasons = {}, {}
    for profile in _club(game, player.team, key):
        if profile.get("player_id") != player.player_id:
            continue
        if family and profile.get("play_family") != family:
            continue
        season = int(profile.get("source_season") or 0)
        if season > player.season:
            continue
        for split in profile.get("splits", []):
            look = split.get("look") or split.get("coverage")
            if look and season >= seasons.get(look, -1):
                out[look] = {**split, "source_season": season}
                seasons[look] = season
    return out


def _coverage_response(
    splits: dict,
    defense: dict,
    position: str,
    response_metric: str | None = None,
    *,
    look_rates: dict | None = None,
) -> dict:
    metric = response_metric or ("yards_per_attempt" if position == "QB" else "yards_per_target")
    sample = (
        "carries"
        if metric == "yards_per_carry"
        else ("attempts" if position == "QB" else "targets")
    )

    def response(split: dict):
        if metric in {"passing_td_rate", "interception_rate"}:
            numerator = split.get("passing_tds" if metric == "passing_td_rate" else "interceptions")
            attempts = split.get("attempts")
            return numerator / attempts if numerator is not None and attempts else None
        return split.get(metric)

    baseline_split = splits.get("all") or {}
    baseline = response(baseline_split)
    if baseline is None:
        return {"status": "unavailable", "mechanism": "No individual coverage baseline"}
    shells = ("cover_0", "cover_1", "cover_2", "cover_3", "cover_4", "cover_6", "cover_2_man")
    looks = (
        tuple(look_rates)
        if look_rates is not None
        else (shells if any(look in splits for look in shells) else ("man", "zone"))
    )
    contributions, effect, mass = [], 0.0, 0.0
    for look in looks:
        split = splits.get(look) or {}
        frequency = look_rates.get(look) if look_rates is not None else defense.get(f"{look}_rate")
        n, observed = split.get(sample), response(split)
        if frequency is None or observed is None or n is None or n < 3:
            continue
        if baseline_split.get("source_season") is not None and (
            split.get("source_season") != baseline_split["source_season"]
        ):
            # Prior-year look quality cannot be compared with a current-year baseline.
            continue
        frequency = min(max(float(frequency), 0.0), 1.0)
        reliability = float(n) / (float(n) + (48.0 if position == "QB" else 24.0))
        shrunk = baseline + reliability * (observed - baseline)
        effect += frequency * (shrunk - baseline)
        mass += frequency
        contributions.append(
            {
                "look": look,
                "opponent_frequency": frequency,
                "sample": n,
                "observed_response": observed,
                "shrunk_response": round(shrunk, 3),
                "source_season": split["source_season"],
            }
        )
    # Unknown looks contribute baseline; available looks are not rescaled to
    # 100% and tiny splits cannot masquerade as a complete matchup profile.
    if mass > 1.0:
        effect /= mass
    return {
        "status": "observed" if contributions else "unavailable",
        "metric": metric,
        "baseline": baseline,
        "baseline_source_season": baseline_split.get("source_season"),
        "matchup_response": round(baseline + effect, 3),
        "response_delta": round(effect, 3),
        "covered_mass": round(min(mass, 1.0), 3),
        "contributions": contributions,
        "used_to_change_projection": False,
        "mechanism": "Player response weighted by opponent look frequency, shrunk to "
        "the player's all-look baseline; uncovered looks keep that baseline",
    }


def _conditional_responses(splits: dict, defense: dict, position: str, metric: str) -> dict:
    """Separate overlapping diagnostic axes; never add them as independent effects."""
    axes = {
        "safety_structure": {
            look: defense.get(f"{look}_rate") for look in ("single_high", "two_high")
        },
        "defensive_package": {
            look: defense.get(f"{look}_rate") for look in ("base", "nickel", "dime")
        },
    }
    if metric == "yards_per_carry":
        axes["box_count"] = {
            look: defense.get(f"{look}_rate") for look in ("stacked_box", "light_box")
        }
    elif position == "QB":
        for axis, active, other in (
            ("pressure", "pressure", "clean"),
            ("blitz", "blitz", "no_blitz"),
        ):
            rate = defense.get(f"{active}_rate")
            axes[axis] = {active: rate, other: 1 - rate if rate is not None else None}
    return {
        axis: _coverage_response(splits, defense, position, metric, look_rates=rates)
        for axis, rates in axes.items()
    }


def _tracking(game: dict, player) -> dict:
    """Retain dated Next Gen/PFR measures without treating operating style as quality."""
    values, sources = {}, {}
    profiles = _club(game, player.team, "player_scheme")
    if not isinstance(profiles, list):
        return {}
    for profile in sorted(profiles, key=lambda row: row.get("source_season") or 0):
        if profile.get("player_id") != player.player_id:
            continue
        tracking = profile.get("tracking") or {}
        season = tracking.get("season") or profile.get("source_season")
        week = tracking.get("week")
        if not season or season > player.season:
            continue
        if season == player.season and week is not None and week >= player.week:
            continue
        for key, value in tracking.items():
            if key in {"season", "week", "source"} or value is None:
                continue
            values[key] = value
            sources[key] = {
                "source_season": season,
                "through_week": week,
                "source": tracking.get("source"),
                "sample": tracking.get("pressure_dropbacks")
                if key in {"pressure_rate", "pressure_dropbacks"}
                else tracking.get("attempts"),
                "league_rank": (profile.get("tracking_ranks") or {}).get(key),
            }
    return {"values": values, "provenance": sources, "used_to_change_projection": False}


def _window(value: dict, phase: str) -> dict:
    if not isinstance(value, dict):
        return {}
    return (value.get("current") or {}).get(phase) or (value.get("combined") or {}).get(phase) or {}


def _injuries(slate, game: dict, player) -> list[dict]:
    rows = []
    for row in getattr(slate, "injuries", []):
        try:
            valid_week = int(float(row.get("week") or 0)) == player.week
            valid_season = not row.get("season") or int(float(row["season"])) == player.season
        except (ValueError, TypeError):
            continue
        if not valid_week:
            continue
        if not valid_season:
            continue
        team = teams.canonical(row.get("team", ""))
        if team not in {player.team, player.opponent}:
            continue
        status = str(row.get("report_status") or row.get("practice_status") or "")
        if not status or "full participation" in status.lower():
            continue
        if "not injury related" in str(row.get("practice_primary_injury", "")).lower():
            continue
        rows.append(
            {
                "team": team,
                "name": row.get("full_name"),
                "position": row.get("position"),
                "status": status,
                "source": "official injury report",
            }
        )
    # Injury lists from the public snapshot contain ESPN names/positions.
    for team in (player.team, player.opponent):
        for row in _club(game, team, "availability_list"):
            if not isinstance(row, dict):
                continue
            rows.append({"team": team, **row, "source": "MLBMA availability snapshot"})
    return rows


def dossier(slate, player) -> dict:
    game = context_game(slate, player)
    offense, off_sources = _phase(slate, game, player.team, "offense")
    defense, def_sources = _phase(slate, game, player.opponent, "defense")
    local = (getattr(slate, "scheme_profiles", {}) or {}).get(player.team)
    m = player.metrics
    sections = [
        {
            "family": "role",
            "status": "model_input",
            "depth_rank": player.depth_rank,
            "depth_slot": player.depth_slot,
            "continuity": player.role_continuity,
            "history_games": player.history_games,
            "injury_status": player.injury_status,
            "reason": player.role_reason,
            "opportunities": {
                k: m[k]
                for k in (
                    "pass_attempts",
                    "targets",
                    "carries",
                    "rush_attempts",
                    "fg_attempts",
                    "pat_made",
                )
                if k in m
            },
            "used_to_change_projection": True,
            "mechanism": "Present roster and depth slot set the role; historical usage earns "
            "weight within one reconciled team opportunity pool",
        },
        _record(
            "offensive_intent",
            offense,
            off_sources,
            GROUPS["intent"],
            "Neutral pass frequency sets the opportunity path; tempo changes plays, "
            "and trailing/leading changes late-game run/pass allocation",
        ),
        _record(
            "formation",
            offense,
            off_sources,
            GROUPS["formation"],
            "Shotgun/under-center/pistol describe the offense's starting presentation, "
            "without identifying its blocking or route assignment",
        ),
        _record(
            "personnel",
            offense,
            off_sources,
            GROUPS["personnel"],
            "11/12/21/13/22 affect available receivers, blockers and defensive package "
            "responses; aggregate frequency does not establish individual route share",
        ),
        _record(
            "concepts",
            offense,
            off_sources,
            GROUPS["concepts"],
            "Motion, play action, RPO and screens are potential pressure/coverage "
            "answers; success requires the offense's measured response to those looks",
        ),
        _record(
            "opponent_coverage",
            defense,
            def_sources,
            GROUPS["coverage"],
            "Evaluate man/zone and each shell against the player's sampled response; "
            "two-high alone does not establish underneath targets or easy rushing",
        ),
        _record(
            "opponent_front_and_pressure",
            defense,
            def_sources,
            GROUPS["front"],
            "Blitz and pressure are distinct. Light/heavy boxes and base/nickel/dime "
            "describe constraints; pressure can remove attempts before it creates checkdowns",
        ),
        _record(
            "offense_response",
            offense,
            off_sources,
            GROUPS["response"],
            "Efficiency vs blitz, pressure, man/zone and concepts tests whether the "
            "offense actually executes the supposed schematic answer",
        ),
        _record(
            "opponent_response",
            defense,
            def_sources,
            GROUPS["response"],
            "Allowed EPA/success distinguishes a frequently used look from a look "
            "the opponent executes well; frequencies are not quality rankings",
        ),
    ]
    passing = _splits(game, player, "player_scheme", "passing")
    receiving = _splits(game, player, "player_coverage")
    rushing = _splits(game, player, "player_scheme", "rushing")
    splits = passing if player.position == "QB" else receiving
    weighted = _coverage_response(splits, defense, player.position)
    sections.append(
        {
            "family": "individual_look_response",
            "status": "observed" if (splits or rushing) else "unavailable",
            "passing_or_receiving": splits,
            "rushing": rushing,
            "coverage_weighted_response": weighted,
            "conditional_responses": {
                "passing_or_receiving": _conditional_responses(
                    splits,
                    defense,
                    player.position,
                    "yards_per_attempt" if player.position == "QB" else "yards_per_target",
                ),
                "rushing": _conditional_responses(
                    rushing, defense, player.position, "yards_per_carry"
                ),
            },
            "tracking": _tracking(game, player),
            "used_to_change_projection": False,
            "mechanism": "Counts and source seasons accompany every player split; "
            "run direction/guard/tackle/end are run-point proxies. Pressure, blitz, "
            "safety and package comparisons overlap and are not summed. Tracking time "
            "and boxes faced describe operating style, not a quality grade",
        }
    )
    own_run, opp_run = (
        _club(game, player.team, "run_game"),
        _club(game, player.opponent, "run_game"),
    )
    line, opp_line = (
        _club(game, player.team, "line_stats"),
        _club(game, player.opponent, "line_stats"),
    )
    lanes = opp_line.get("defense_run_front") or {}
    lane_pairs = []
    for look, split in rushing.items():
        opponent_lane = lanes.get(look if look.startswith("gap_") else f"lane_{look}")
        if opponent_lane:
            lane_pairs.append({"run_point_proxy": look, "player": split, "opponent": opponent_lane})
    sections.append(
        {
            "family": "run_game_and_trenches",
            "status": "observed" if (line or opp_line or own_run or opp_run) else "unavailable",
            "offense_run": _window(own_run, "offense"),
            "opponent_run": _window(opp_run, "defense"),
            "run_source_seasons": {
                "offense": own_run.get("seasons"),
                "opponent": opp_run.get("seasons"),
            },
            "offensive_line": line.get("offense", {}),
            "opponent_line": opp_line.get("defense", {}),
            "line_source_season": line.get("season"),
            "opponent_line_source_season": opp_line.get("season"),
            "run_point_matchups": lane_pairs,
            "weighted_run_point_response": prop_matchup_paths.run_points(
                rushing, lanes, _window(opp_run, "defense")
            ),
            "used_to_change_projection": False,
            "mechanism": "Compare stuff, explosives, success, yards before contact "
            "and sack/hit rates; do not relabel a run gap as blocking scheme",
        }
    )
    own_rz, opp_rz = _club(game, player.team, "red_zone"), _club(game, player.opponent, "red_zone")
    sections.append(
        {
            "family": "drive_finishing",
            "status": "observed" if (own_rz or opp_rz) else "unavailable",
            "offense": _window(own_rz, "offense"),
            "opponent": _window(opp_rz, "defense"),
            "player_red_zone": [
                r
                for r in (own_rz.get("current") or {}).get("players", [])
                if r.get("player_id") == player.player_id
            ],
            "used_to_change_projection": False,
            "mechanism": "Drives reaching scoring range create chances. Touchdown "
            "conversion removes field goals but adds PATs; empty drives "
            "and fourth-down attempts do not guarantee a kick",
            "kicking_path": prop_matchup_paths.kicking(slate, game, player)
            if player.position == "K"
            else {},
        }
    )
    injuries = _injuries(slate, game, player)
    sections.append(
        {
            "family": "availability",
            "status": "observed" if injuries else "unavailable",
            "reports": injuries,
            "mechanism": "OL absences can disrupt protection/run efficiency; "
            "skill-player absences redistribute a fixed pool. Verify "
            "inactive lists and route/workload replacement before kickoff",
        }
    )
    sections.append(
        {
            "family": "regime",
            "status": "observed" if local else "unavailable",
            "flags": list(getattr(local, "regime_flags", ())),
            "reaction_weight": getattr(local, "reaction_weight", None),
            "reaction_window": getattr(local, "reaction_window", None),
            "staff_continuity": getattr(local, "staff_continuity", None),
            "mechanism": "Recent coordinator/QB/play-call changes widen uncertainty; "
            "do not assume prior-year shells apply unchanged",
        }
    )
    schedule = next((row for row in slate.games if row.get("game_id") == player.game_id), {})
    sections.append(
        {
            "family": "game_environment",
            "status": "context",
            "implied_team_points": player.implied_team_points,
            "team_environment_source": player.team_environment_source,
            "roof": game.get("roof") or schedule.get("roof"),
            "surface": schedule.get("surface"),
            "venue": game.get("venue"),
            "rest_days": game.get(f"{_side(game, player.team)}_rest_days"),
            "travel_km": game.get(f"{_side(game, player.team)}_travel_km"),
            "wind_forecast": None,
            "mechanism": "Scoring environment supports drive volume; roof, rest and "
            "travel qualify it. Wind has no adjustment without a dated forecast",
        }
    )
    sections.append(
        {
            "family": "unobserved_assignments",
            "status": "unavailable",
            "fields": list(MISSING_ASSIGNMENTS),
            "mechanism": "These details require additional charting; no guessed "
            "route, blocking, shadow or kicker-distance claim is scored",
        }
    )
    return {
        "player_id": player.player_id,
        "sections": sections,
        "coverage_response": weighted,
        "offense": offense,
        "defense": defense,
        "injuries": injuries,
        "external_context_available": bool(game),
        "source_snapshot": (getattr(slate, "prop_context", {}) or {}).get("generated_at_utc"),
        "projection_scheme_inputs": player.scheme_context,
    }


def explain(player, metric: str, line: float, side: str, scouting: dict) -> dict:
    """A concrete opportunity hurdle, supporting evidence and failure paths."""
    m, ctx = player.metrics, player.scheme_context or {}
    over = side == "OVER"
    word = "exceed" if over else "stay below"
    passages = [
        f"{player.player_name} must {word} {line:g} {metric.replace('_', ' ')}; "
        f"the role-aware projection is {m.get(metric, 0):.1f}."
    ]
    requirements = {}
    if metric == "passing_yards" and m.get("pass_attempts", 0) > 0:
        attempts = m.get("pass_attempts", 0)
        requirements["yards_per_attempt_at_projected_volume"] = round(line / attempts, 2)
        passages.append(
            f"At {attempts:.1f} attempts the hurdle is {line / attempts:.2f} "
            "yards per attempt; sacks remove attempts and pressure tests that efficiency."
        )
    elif metric in {"completions", "passing_tds", "interceptions"} and m.get("pass_attempts", 0):
        attempts = m["pass_attempts"]
        needed = math.floor(line) + 1 if over else max(math.ceil(line) - 1, 0)
        rate = needed / attempts
        requirements["minimum_count" if over else "maximum_count"] = needed
        requirements["rate_per_attempt_at_projected_volume"] = round(rate, 4)
        passages.append(
            f"At {attempts:.1f} attempts, {'at least' if over else 'at most'} "
            f"{needed} {metric.replace('_', ' ')} represents {rate:.1%} "
            "per attempt; drive volume and the player's observed look-specific "
            "completion/TD/interception rate must sustain that path."
        )
    elif metric == "rushing_yards":
        carries = m.get("carries", m.get("rush_attempts", 0))
        if carries:
            requirements["yards_per_carry_at_projected_volume"] = round(line / carries, 2)
            passages.append(
                f"At {carries:.1f} carries the hurdle is {line / carries:.2f} "
                "yards per carry; stuffs and loss of late-game carries threaten it."
            )
    elif metric in {"receiving_yards", "receptions"}:
        targets = m.get("targets", 0)
        if targets:
            key = "yards_per_target" if metric == "receiving_yards" else "catch_rate"
            needed = (
                line
                if metric == "receiving_yards"
                else (math.floor(line) + 1 if over else max(math.ceil(line) - 1, 0))
            )
            requirements[f"{key}_at_projected_volume"] = round(needed / targets, 3)
            shown = (
                f"{needed / targets:.2f} yards per target"
                if metric == "receiving_yards"
                else (
                    f"{'at least' if over else 'at most'} {needed} "
                    f"{'catch' if needed == 1 else 'catches'} "
                    f"({needed / targets:.0%} of projected targets)"
                )
            )
            passages.append(
                f"At {targets:.1f} targets the hurdle is {shown}; coverage "
                "response must survive pressure and competition for the team target pool."
            )
    elif metric == "rush_receiving_yards":
        passages.append(
            f"Rushing {m.get('rushing_yards', 0):.1f} plus receiving "
            f"{m.get('receiving_yards', 0):.1f}; a receiving path can offset lost "
            "carries only if the back retains passing-down routes. Dependence is bounded."
        )
    elif metric == "pass_rush_yards":
        passages.append(
            "Passing and rushing share game script; pressure can create scrambles "
            "or sacks. The total is bounded without assuming independent paths."
        )
    elif metric in {"fg_made", "pat_made", "kicking_points"}:
        requirements.update(
            {
                "field_goal_attempts": m.get("fg_attempts"),
                "field_goals_made": m.get("fg_made"),
                "pats_made": m.get("pat_made"),
            }
        )
        passages.append(
            f"Projection: {m.get('fg_attempts', 0):.1f} FG attempts, "
            f"{m.get('fg_made', 0):.1f} makes and {m.get('pat_made', 0):.1f} PATs. "
            "Evaluate scoring-range trips and stalled drives, not just team points."
        )
    if metric in {"fg_made", "pat_made", "kicking_points"}:
        requirements["minimum_count" if over else "maximum_count"] = (
            math.floor(line) + 1 if over else max(math.ceil(line) - 1, 0)
        )
    offense, defense = scouting["offense"], scouting["defense"]
    intent = []
    for key, label in (
        ("neutral_pass_rate", "neutral pass"),
        ("formation_shotgun_rate", "shotgun"),
        ("personnel_11_rate", "11 personnel"),
        ("motion_rate", "motion"),
        ("play_action_rate", "play action"),
        ("screen_rate", "screens"),
    ):
        if key in offense:
            intent.append(f"{label} {offense[key]:.0%}")
    if intent:
        passages.append(f"{player.team} offensive intent: " + ", ".join(intent) + ".")
    opponent = []
    for key, label in (
        ("man_rate", "man"),
        ("zone_rate", "zone"),
        ("two_high_rate", "two-high"),
        ("blitz_rate", "blitz"),
        ("pressure_rate", "pressure"),
        ("stacked_box_rate", "heavy box"),
    ):
        if key in defense:
            opponent.append(f"{label} {defense[key]:.0%}")
    if opponent:
        passages.append(f"{player.opponent} observed defense: " + ", ".join(opponent) + ".")
    individual = next(s for s in scouting["sections"] if s["family"] == "individual_look_response")
    response_metric = {
        "passing_yards": "yards_per_attempt",
        "receiving_yards": "yards_per_target",
        "receptions": "catch_rate",
        "completions": "completion_rate",
        "passing_tds": "passing_td_rate",
        "interceptions": "interception_rate",
    }.get(metric)
    weighted = (
        _coverage_response(
            individual["passing_or_receiving"], defense, player.position, response_metric
        )
        if response_metric
        else {"status": "unavailable"}
    )
    signals, conflicts = [], []
    if weighted.get("status") == "observed":
        delta = weighted["response_delta"]
        baseline_label = (
            f"{weighted['baseline']:.1%}"
            if response_metric.endswith("rate")
            else f"{weighted['baseline']:.2f}"
        )
        matchup_label = (
            f"{weighted['matchup_response']:.1%}"
            if response_metric.endswith("rate")
            else f"{weighted['matchup_response']:.2f}"
        )
        passages.append(
            f"Individual sampled {weighted['metric'].replace('_', ' ')}: "
            f"{baseline_label} baseline, {matchup_label} "
            f"under sampled opponent looks ({weighted['covered_mass']:.0%} covered)."
        )
        floor = (
            0.003
            if metric in {"passing_tds", "interceptions"}
            else (0.01 if metric in {"receptions", "completions"} else 0.05)
        )
        if abs(delta) >= floor:
            target = signals if (delta > 0) == over else conflicts
            response_label = {
                "interceptions": "interception frequency",
                "passing_tds": "touchdown frequency",
                "receptions": "catch rate",
                "completions": "completion rate",
            }.get(metric, "yardage efficiency")
            target.append(
                "Individual look response supports "
                + ("higher" if delta > 0 else "lower")
                + " "
                + response_label
            )
    conditionals = {}
    conditional_metric = "yards_per_carry" if metric == "rushing_yards" else response_metric
    if conditional_metric:
        relevant_splits = (
            individual["rushing"]
            if metric == "rushing_yards"
            else (individual["passing_or_receiving"])
        )
        conditionals = _conditional_responses(
            relevant_splits, defense, player.position, conditional_metric
        )
        floor = (
            0.003
            if metric in {"passing_tds", "interceptions"}
            else (0.01 if metric in {"receptions", "completions"} else 0.05)
        )
        for axis, response in conditionals.items():
            if response["status"] != "observed":
                continue
            delta = response["response_delta"]
            scale = ".1%" if conditional_metric.endswith("rate") else ".2f"
            note = (
                f"Individual {axis.replace('_', ' ')} response: "
                f"{format(response['matchup_response'], scale)} "
                f"{conditional_metric.replace('_', ' ')} versus "
                f"{format(response['baseline'], scale)} baseline; "
                f"{response['covered_mass']:.0%} of this axis sampled."
            )
            passages.append(note)
            if abs(delta) >= floor:
                (signals if (delta > 0) == over else conflicts).append(note)
    tracking = (individual.get("tracking") or {}).get("values") or {}
    if player.position == "QB" and tracking.get("avg_time_to_throw") is not None:
        passages.append(
            f"Tracked time to throw: {tracking['avg_time_to_throw']:.2f}s; "
            "interpret with this opponent's pressure and the sampled clean/pressure "
            "response, without assuming longer holding time causes sacks."
        )
    if player.position == "RB" and metric in {"rushing_yards", "rush_receiving_yards"}:
        tracked = [
            f"{key.replace('_', ' ')} {tracking[key]:.2f}"
            for key in ("expected_yards_per_carry", "ryoe_per_carry", "avg_time_to_los")
            if key in tracking
        ]
        if tracking.get("eight_plus_box_rate") is not None:
            tracked.append(f"eight-plus boxes faced {tracking['eight_plus_box_rate']:.0%}")
        if tracked:
            passages.append(
                "Individual tracking: " + ", ".join(tracked) + ". "
                "These are observed execution/style measures; future box assignments "
                "and blocking schemes remain unknown."
            )
    if metric in {
        "passing_yards",
        "pass_attempts",
        "completions",
        "receiving_yards",
        "receptions",
        "passing_tds",
        "interceptions",
    }:
        effects = [("pass_attempt_delta", "Scheme pass volume")]
        if metric in {"passing_yards", "receiving_yards"}:
            effects.append(("pass_efficiency_delta", "Scheme pass efficiency"))
    elif metric in {"rushing_yards", "carries", "rush_attempts"} and player.position == "RB":
        effects = [("carry_delta", "Scheme run volume")]
        if metric == "rushing_yards":
            effects.append(("rush_efficiency_delta", "Scheme run efficiency"))
    else:
        effects = []
    for key, label in effects:
        value = ctx.get(key)
        if value is not None and abs(value) > 0.05:
            (signals if (value > 0) == over else conflicts).append(f"{label} {value:+.2f}")
    if player.position in {"WR", "RB"} and metric in {"receiving_yards", "receptions"}:
        multiplier = float(ctx.get("target_multiplier", 1.0))
        if abs(multiplier - 1) > 0.02:
            (signals if (multiplier > 1) == over else conflicts).append(
                f"Position target allocation ×{multiplier:.2f}"
            )
    if metric in {"fg_made", "pat_made", "kicking_points"}:
        finishing = next(s for s in scouting["sections"] if s["family"] == "drive_finishing")
        kick_support, kick_conflicts, note = prop_matchup_paths.kicking_signals(
            metric, over, finishing.get("kicking_path") or {}
        )
        signals.extend(kick_support)
        conflicts.extend(kick_conflicts)
        if note:
            passages.append(note)
        for club, key in ((player.team, "offense"), (player.opponent, "opponent")):
            rz = finishing[key]
            if rz:
                passages.append(
                    f"{club} scoring-range evidence: {rz.get('trips', 0)} trips "
                    f"in {rz.get('games', 0)} games; TD conversion "
                    f"{rz.get('td_rate', 0):.0%}, scoring conversion "
                    f"{rz.get('score_rate', 0):.0%}."
                )
    if metric in {"rushing_yards", "rush_receiving_yards"} and player.position == "RB":
        run = next(s for s in scouting["sections"] if s["family"] == "run_game_and_trenches")
        weighted_run = run["weighted_run_point_response"]
        if weighted_run.get("status") == "proxy":
            passages.append(
                f"Run-point mix maps to opponent yield "
                f"{weighted_run['matched_run_point_yards_per_carry']:.2f} vs "
                f"{weighted_run['opponent_baseline_yards_per_carry']:.2f} overall YPC "
                f"({weighted_run['covered_mass']:.0%} carry distribution covered)."
            )
            delta = weighted_run["response_delta"]
            if abs(delta) > 0.10:
                (signals if (delta > 0) == over else conflicts).append(
                    "Run-point proxy matchup supports "
                    + ("higher" if delta > 0 else "lower")
                    + " rushing efficiency"
                )
    risk = [
        "Leading/trailing game script changes opportunities",
        "Recheck starters, inactive lists and the actual posted threshold",
    ]
    if player.position in {"WR", "RB"}:
        risk.append(
            "Route participation, passing-down usage and double-team assignments are uncharted"
        )
    if player.position == "QB":
        risk.append("Protection failures or early exit can overwhelm favorable coverage")
    if player.position == "K":
        risk += [
            "TDs replace field-goal chances; fourth-down aggression and empty drives remove kicks",
            "Distance mix and dated wind are unavailable; make-rate and Poisson "
            "assumptions need testing",
        ]
    own_injuries = [r for r in scouting["injuries"] if r.get("team") == player.team]
    if own_injuries:
        risk.append(
            "Offensive availability: "
            + "; ".join(
                f"{r.get('position', '')} {r.get('name', '')} ({r.get('status', '')})"
                for r in own_injuries[:6]
            )
        )
    if player.injury_status:
        risk.append(f"Player designation: {player.injury_status}")
    if not scouting["external_context_available"]:
        risk.append("Dated MLBMA player/trench/red-zone evidence unavailable for this fixture")
    passages.append(f"Role: {player.depth_slot}; {player.role_reason}")
    return {
        "thesis": " ".join(passages),
        "threshold_requirements": requirements,
        "support": signals,
        "conflicts": conflicts,
        "failure_paths": risk,
        "schematic_assessment": "mixed"
        if conflicts
        else "supported"
        if signals
        else "neutral observed matchup"
        if scouting["external_context_available"]
        else "role-based; individual scheme support incomplete",
        "evidence": [
            {
                **section,
                "selected_market_responses": {
                    "metric": metric,
                    "coverage": weighted,
                    "conditional_axes": conditionals,
                },
            }
            if section["family"] == "individual_look_response"
            else section
            for section in scouting["sections"]
        ],
    }


def player_unavailable(player, scouting: dict) -> bool:
    statuses = {"out", "doubtful", "inactive", "injured reserve", "suspended"}
    if str(player.injury_status or "").lower() in statuses:
        return True
    if str(player.roster_status or "").lower() in {"inactive", "ir", "injured reserve"}:
        return True
    return any(
        r.get("team") == player.team
        and normalise(r.get("name") or r.get("full_name") or "") == normalise(player.player_name)
        and str(r.get("status", "")).lower() in statuses
        for r in scouting["injuries"]
    )
