"""Pre-forecast player process, role stability and regression evidence.

Observed opportunity, completion expectation and play design are separate from
the outcome they help predict. Missing charting stays missing. Participation
from the forecast season is never used: nflverse releases it after the season.
The fitted correction/uncertainty artifact is versioned independently of prices.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from . import teams
from .sources.nflverse import number

VERSION = "nfl-player-intelligence/1.0.0"
RECENT_GAMES = 3


def order(season: int, week: int) -> int:
    return season * 24 + week


def val(row: dict, key: str) -> float:
    return float(number(row.get(key)) or 0.0)


def flag(value) -> bool | None:
    if str(value).lower() in {"true", "t", "yes"}:
        return True
    if str(value).lower() in {"false", "f", "no"}:
        return False
    parsed = number(value)
    return None if parsed is None else bool(parsed)


def play_key(row: dict, game: str = "game_id", play: str = "play_id") -> tuple[str, str]:
    return str(row.get(game) or ""), str(row.get(play) or "").split(".")[0]


def process_games(
    pbp_rows: list[dict],
    *,
    season: int,
    week: int,
    participation_rows: list[dict] | None = None,
    charting_rows: list[dict] | None = None,
) -> dict[tuple[int, str, str], dict[str, float]]:
    """Per-player/game opportunity and process; every play is strictly pre-week."""
    looks = {}
    for row in participation_rows or []:
        key = play_key(row, "nflverse_game_id")
        try:
            source_season = int(key[0][:4])
        except ValueError:
            continue
        if source_season >= season:
            continue
        kind = str(row.get("defense_man_zone_type") or "").lower()
        if "man" in kind:
            looks[key] = "man"
        elif "zone" in kind:
            looks[key] = "zone"
    chart = {
        play_key(row, "nflverse_game_id", "nflverse_play_id"): row for row in charting_rows or []
    }
    out = defaultdict(lambda: defaultdict(float))
    seen = set()
    target = order(season, week)
    for row in pbp_rows:
        row_season, row_week = int(val(row, "season")), int(val(row, "week"))
        point = order(row_season, row_week)
        key = play_key(row)
        if (
            not row_season
            or not row_week
            or point >= target
            or key in seen
            or str(row.get("season_type") or "REG") != "REG"
            or val(row, "no_play")
            or row.get("play_type") == "no_play"
        ):
            continue
        seen.add(key)
        team = teams.canonical(row.get("posteam") or "")
        if not team:
            continue
        yardline = number(row.get("yardline_100"))
        endzone = yardline is not None and val(row, "air_yards") >= yardline
        look = looks.get(key)
        design = chart.get(key, {})
        situations = []
        if look:
            situations.append(look)
        if number(design.get("n_blitzers")) is not None:
            situations.append("blitz" if val(design, "n_blitzers") > 0 else "no_blitz")
        screen = flag(design.get("is_screen_pass"))
        if screen is not None:
            situations.append("screen" if screen else "no_screen")
        action = flag(design.get("is_play_action"))
        if action is not None:
            situations.append("play_action" if action else "no_play_action")
        passer = str(row.get("passer_player_id") or "")
        receiver = str(row.get("receiver_player_id") or "")
        passing = bool(
            val(row, "pass_attempt") and not val(row, "sack") and not val(row, "qb_spike")
        )
        if passing:
            cp = number(row.get("cp"))
            air, xyac = number(row.get("air_yards")), number(row.get("xyac_mean_yardage"))
            expected = (
                max(0.0, cp * (air + xyac))
                if cp is not None and air is not None and xyac is not None
                else None
            )
            for pid, prefix in ((passer, "pass"), (receiver, "receive")):
                if not pid:
                    continue
                cell = out[(point, team, pid)]
                yards = val(row, "passing_yards" if prefix == "pass" else "receiving_yards")
                cell[f"{prefix}_opps"] += 1
                cell[f"{prefix}_yards"] += yards
                cell[f"{prefix}_completions"] += val(row, "complete_pass")
                if cp is not None and 0 <= cp <= 1:
                    cell[f"{prefix}_cp_opps"] += 1
                    cell[f"{prefix}_expected_catches"] += cp
                    cell[f"{prefix}_cp_completions"] += val(row, "complete_pass")
                if expected is not None:
                    cell[f"{prefix}_expected_yards"] += expected
                    cell[f"{prefix}_expected_yards_actual"] += yards
                    cell[f"{prefix}_expected_yards_opps"] += 1
                cell[f"{prefix}_explosive_yards"] += max(0.0, yards - 25.0)
                cell[f"{prefix}_inside10"] += yardline is not None and yardline <= 10
                cell[f"{prefix}_endzone"] += endzone
                cell[f"{prefix}_tds"] += val(row, "pass_touchdown")
                for situation in situations:
                    cell[f"{prefix}_{situation}_opps"] += 1
                    cell[f"{prefix}_{situation}_yards"] += yards
        rusher = str(row.get("rusher_player_id") or "")
        if rusher and val(row, "rush_attempt") and not val(row, "qb_kneel"):
            cell = out[(point, team, rusher)]
            yards = val(row, "rushing_yards")
            cell["rush_opps"] += 1
            cell["rush_yards"] += yards
            cell["rush_core_yards"] += min(15.0, max(-5.0, yards))
            cell["rush_explosive_yards"] += max(0.0, yards - 15.0)
            cell["rush_inside10"] += yardline is not None and yardline <= 10
            cell["rush_inside5"] += yardline is not None and yardline <= 5
            cell["rush_scrambles"] += val(row, "qb_scramble")
            cell["rush_tds"] += val(row, "rush_touchdown")
    return {key: dict(cell) for key, cell in out.items()}


def history_index(rows: list[dict], *, season: int, week: int) -> dict[str, list[dict]]:
    indexed = defaultdict(list)
    target = order(season, week)
    seen = set()
    for row in rows:
        point = order(int(val(row, "season")), int(val(row, "week")))
        pid = str(row.get("player_id") or "")
        team = teams.canonical(row.get("team") or "")
        key = (point, team, pid)
        if (
            not pid
            or not team
            or point >= target
            or key in seen
            or str(row.get("season_type") or "REG") != "REG"
        ):
            continue
        seen.add(key)
        indexed[pid].append({**row, "_order": point, "_team": team})
    return {pid: sorted(games, key=lambda row: row["_order"]) for pid, games in indexed.items()}


def feature_profile(
    player, histories: dict, process: dict, *, matchup=None, team_totals: dict | None = None
) -> tuple[dict[str, float], dict]:
    """Inputs contain no current-game outcomes or player-prop lines."""
    target = order(player.season, player.week)
    rows = histories.get(player.player_id, [])
    last_other = max((r["_order"] for r in rows if r["_team"] != player.team), default=-1)
    stint = [r for r in rows if r["_team"] == player.team and r["_order"] > last_other]
    recent = stint[-RECENT_GAMES:]
    role_field = (
        "carries"
        if player.position == "RB"
        else ("attempts" if player.position == "QB" else "targets")
    )

    def weighted(items, field, half_life=12.0):
        return sum(val(row, field) * 0.5 ** ((target - row["_order"]) / half_life) for row in items)

    def rate(items, numerator, denominator, prior, pseudo=0.0):
        den = weighted(items, denominator)
        return (weighted(items, numerator) + prior * pseudo) / max(den + pseudo, 1e-9)

    weighted_process, recent_process = defaultdict(float), defaultdict(float)
    for row in rows:
        cell = process.get((row["_order"], row["_team"], player.player_id), {})
        weight = 0.5 ** ((target - row["_order"]) / 12.0)
        for key, value in cell.items():
            weighted_process[key] += weight * value
            if row in recent:
                recent_process[key] += value

    def process_rate(cell, num, den, default=0.0):
        return cell[num] / cell[den] if cell[den] else default

    features = {
        "intercept": 1.0,
        "log_history": math.log1p(len(stint)),
        "new_team": float(bool(rows) and len(stint) < 3),
        "questionable": float(
            str(player.injury_status or "").lower() in {"questionable", "doubtful"}
        ),
    }
    metrics = player.metrics
    role_values = [val(row, role_field) for row in stint[-6:]]
    mean_role = sum(role_values) / max(len(role_values), 1)
    role_sd = math.sqrt(sum((v - mean_role) ** 2 for v in role_values) / max(len(role_values), 1))
    features["role_cv"] = min(3.0, role_sd / max(mean_role, 1.0))
    features["zero_role_rate"] = (
        sum(v == 0 for v in role_values) / len(role_values) if role_values else 0.0
    )
    shares = []
    for row in stint:
        total = (team_totals or {}).get((row["_order"], player.team), {}).get(role_field)
        if total:
            shares.append((row["_order"], val(row, role_field) / total))
    if shares:
        weights = [0.5 ** ((target - point) / 6.0) for point, _ in shares]
        long_share = sum(w * value for w, (_, value) in zip(weights, shares)) / sum(weights)
        recent_share = sum(v for _, v in shares[-3:]) / len(shares[-3:])
        reliability = len(shares) / (len(shares) + 3.0)
        features["role_trend"] = (
            max(-1.0, min(1.0, math.log((recent_share + 0.02) / (long_share + 0.02)))) * reliability
        )
    else:
        long_share = recent_share = None
        features["role_trend"] = 0.0
    features["depth"] = float(player.depth_rank - 1)
    features["log_implied"] = math.log(max(float(player.implied_team_points or 22.5), 7) / 22.5)
    prefix = "pass" if player.position == "QB" else "receive"
    volume_key = "pass_attempts" if player.position == "QB" else "targets"
    base_volume = metrics.get(volume_key, 0.0)
    yard_key = "passing_yards" if player.position == "QB" else "receiving_yards"
    base_rate = metrics.get(yard_key, 0.0) / max(base_volume, 0.1)
    denominator = "attempts" if player.position == "QB" else "targets"
    numerator = "passing_yards" if player.position == "QB" else "receiving_yards"
    catch_num = "completions" if player.position == "QB" else "receptions"
    cp_n = weighted_process[f"{prefix}_cp_opps"]
    cp_reliability = cp_n / (cp_n + 60.0)
    expected_catch = process_rate(
        weighted_process, f"{prefix}_expected_catches", f"{prefix}_cp_opps"
    )
    actual_catch = process_rate(
        weighted_process, f"{prefix}_cp_completions", f"{prefix}_cp_opps", expected_catch
    )
    features["completion_over_expectation"] = (actual_catch - expected_catch) * cp_reliability
    expected_n = weighted_process[f"{prefix}_expected_yards_opps"]
    expected_rate = process_rate(
        weighted_process, f"{prefix}_expected_yards", f"{prefix}_expected_yards_opps", base_rate
    )
    features["process_rate_gap"] = (expected_rate - base_rate) * expected_n / (expected_n + 60)
    features["recent_efficiency"] = max(
        -5.0, min(5.0, rate(recent, numerator, denominator, base_rate, 15) - base_rate)
    )
    base_catch = metrics.get(catch_num, 0.0) / max(base_volume, 0.1)
    features["expected_catch_gap"] = (expected_catch - base_catch) * cp_reliability
    features["recent_catch_gap"] = rate(recent, catch_num, denominator, base_catch, 15) - base_catch
    features["explosive_dependence"] = process_rate(
        weighted_process, f"{prefix}_explosive_yards", f"{prefix}_yards"
    )
    features["inside10_rate"] = process_rate(
        weighted_process, f"{prefix}_inside10", f"{prefix}_opps"
    )
    features["endzone_rate"] = process_rate(weighted_process, f"{prefix}_endzone", f"{prefix}_opps")
    rushes = weighted_process["rush_opps"]
    rush_yards = process_rate(weighted_process, "rush_yards", "rush_opps", 4.3)
    features["rush_core_gap"] = (
        (process_rate(weighted_process, "rush_core_yards", "rush_opps", rush_yards) - rush_yards)
        * rushes
        / (rushes + 45)
    )
    features["rush_explosive_dependence"] = process_rate(
        weighted_process, "rush_explosive_yards", "rush_yards"
    )
    features["rush_inside5_rate"] = process_rate(weighted_process, "rush_inside5", "rush_opps")
    features["rush_inside10_rate"] = process_rate(weighted_process, "rush_inside10", "rush_opps")
    features["scramble_share"] = process_rate(weighted_process, "rush_scrambles", "rush_opps")
    features["recent_rush_efficiency"] = max(
        -4.0, min(4.0, rate(recent, "rushing_yards", "carries", rush_yards, 20) - rush_yards)
    )
    look_evidence = {}
    for look, opposite, probability in (
        ("man", "zone", getattr(matchup, "expected_man_rate", None)),
        ("blitz", "no_blitz", getattr(matchup, "expected_blitz_rate", None)),
        ("screen", "no_screen", None),
        ("play_action", "no_play_action", getattr(matchup, "expected_play_action_rate", None)),
    ):
        n1, n0 = (weighted_process[f"{prefix}_{name}_opps"] for name in (look, opposite))
        r1 = process_rate(
            weighted_process, f"{prefix}_{look}_yards", f"{prefix}_{look}_opps", base_rate
        )
        r0 = process_rate(
            weighted_process, f"{prefix}_{opposite}_yards", f"{prefix}_{opposite}_opps", base_rate
        )
        observed_mix = n1 / (n1 + n0) if n1 + n0 else 0.0
        # Missing expected opponent tendency cannot be replaced with an invented mix.
        delta = (
            0.0
            if probability is None
            else (float(probability) - observed_mix) * (r1 - r0) * min(n1, n0) / (min(n1, n0) + 30)
        )
        features[f"{look}_response"] = max(-3.0, min(3.0, delta))
        if n1 + n0:
            look_evidence[look] = {
                "weighted_opportunities": round(n1 + n0, 1),
                "historical_mix": round(observed_mix, 4),
                "expected_mix": probability,
                "shrunk_rate_delta": round(delta, 4),
            }
    warnings = []
    last_stint_order = max((r["_order"] for r in stint), default=None)
    missed_games = (
        sum(
            point > last_stint_order
            for point, team in (team_totals or {})
            if team == player.team and point < target
        )
        if last_stint_order is not None
        else 0
    )
    if missed_games >= 2:
        warnings.append(
            f"No recorded workload in the last {missed_games} team games; "
            "the returning role remains uncertain"
        )
    if len(stint) < 4:
        warnings.append("Limited evidence in the current team role")
    if features["role_cv"] > 0.5:
        warnings.append("Recent opportunity is volatile")
    if not cp_n and player.position != "K":
        warnings.append("Completion-expectation evidence unavailable")
    if features["questionable"]:
        warnings.append("Forecast is conditional on playing; injury availability is unresolved")
    evidence = {
        "version": VERSION,
        "current_stint_games": len(stint),
        "recent_games": len(recent),
        "recent_opportunity_mean": round(mean_role, 3),
        "stability_window_games": len(role_values),
        "recent_workload_gap_team_games": missed_games,
        "recent_opportunity_cv": round(features["role_cv"], 3),
        "recent_share": recent_share,
        "baseline_share": long_share,
        "completion_expectation_opportunities": round(cp_n, 1),
        "expected_completion_rate": round(expected_catch, 4) if cp_n else None,
        "observed_completion_rate": round(actual_catch, 4) if cp_n else None,
        "process_yards_per_opportunity": round(expected_rate, 3) if expected_n else None,
        "individual_look_response": look_evidence,
        "warnings": warnings,
        "cutoff": {"season": player.season, "week_exclusive": player.week},
        "participation_policy": "previous seasons only; current-season charting stays separate",
        "recent_metric_means": {
            metric: round(sum(val(r, field) for r in recent) / len(recent), 3)
            for metric, field in {
                "targets": "targets",
                "carries": "carries",
                "rush_attempts": "carries",
                "pass_attempts": "attempts",
                "receptions": "receptions",
                "completions": "completions",
                "passing_yards": "passing_yards",
                "rushing_yards": "rushing_yards",
                "receiving_yards": "receiving_yards",
                "passing_tds": "passing_tds",
                "interceptions": "passing_interceptions",
            }.items()
            if recent and metric in metrics
        },
    }
    return features, evidence


def team_totals(rows: list[dict], *, season: int, week: int) -> dict:
    target = order(season, week)
    out = defaultdict(lambda: defaultdict(float))
    for row in rows:
        point = order(int(val(row, "season")), int(val(row, "week")))
        if point >= target or str(row.get("season_type") or "REG") != "REG":
            continue
        key = (point, teams.canonical(row.get("team") or ""))
        for metric in ("attempts", "targets", "carries", "completions", "passing_yards"):
            out[key][metric] += val(row, metric)
    return dict(out)


def add_quarterback_features(rows: list[dict], totals: dict, *, season: int, week: int) -> None:
    """Anticipated passer quality relative to the receiver's team history.

    A receiver does not inherit his previous quarterback's accuracy unchanged.
    Effects are learned out of time; this function supplies measured inputs only.
    """
    quarterbacks = {(r["game_id"], r["team"]): r for r in rows if r["position"] == "QB"}
    target = order(season, week)
    past = defaultdict(lambda: defaultdict(float))
    for (point, team), stats in totals.items():
        if point >= target:
            continue
        weight = 0.5 ** ((target - point) / 4)
        for metric in ("attempts", "completions", "passing_yards"):
            past[team][metric] += weight * stats.get(metric, 0)
    for row in rows:
        qb = quarterbacks.get((row["game_id"], row["team"]))
        history = past[row["team"]]
        previous_attempts = history["attempts"]
        next_attempts = qb["metrics"].get("pass_attempts", 0) if qb else 0
        if next_attempts > 0 and previous_attempts > 0:
            completion_shift = (
                qb["metrics"]["completions"] / next_attempts
                - history["completions"] / previous_attempts
            )
            yards_shift = (
                qb["metrics"]["passing_yards"] / next_attempts
                - history["passing_yards"] / previous_attempts
            )
        else:
            completion_shift = yards_shift = 0.0
        row["features"].update(
            {"qb_completion_shift": completion_shift, "qb_yards_shift": yards_shift}
        )
        row.get("evidence", {})["quarterback_context"] = {
            "starter_id": qb["player_id"] if qb else None,
            "completion_rate_shift": round(completion_shift, 4),
            "yards_per_attempt_shift": round(yards_shift, 4),
            "status": "measured model input; coefficient acceptance is reported separately",
        }


@lru_cache(maxsize=1)
def parameters() -> dict:
    path = Path(__file__).with_name("prop_parameters.json")
    if not path.exists():
        return {"corrections": {}, "uncertainty": {}, "state": "not_fitted"}
    return json.loads(path.read_text(encoding="utf-8"))


def adjustment(spec: dict | None, features: dict) -> float:
    if not spec or not spec.get("enabled", True):
        return 0.0
    total = float(spec.get("intercept", 0.0))
    for name, center, scale, coefficient in zip(
        spec["features"],
        spec["center"],
        spec["scale"],
        spec["coefficients"],
    ):
        value = features.get(name)
        if value is None and name == "completion_over_expectation":
            value = features.get("completion_luck", 0.0)  # old research table only
        total += float(coefficient) * (float(value or 0) - center) / max(scale, 1e-8)
    limit = float(spec.get("limit", 0.5))
    return max(-limit, min(limit, total))


def correct_metrics(rows: list[dict], params: dict | None = None) -> list[dict]:
    """Apply fitted role/rate corrections, retaining team opportunity budgets.

    The same function runs in the historical audit and in production. A new
    player-prop line cannot enter this API. Coefficients are accepted on a later
    season, and the artifact dates that decision separately from the final fit.
    """
    params = parameters() if params is None else params
    specs = params.get("corrections", {})
    revised = []
    for row in rows:
        base, features, position = row["metrics"], row["features"], row["position"]
        metrics = dict(base)
        for metric in ("targets", "carries", "pass_attempts", "rush_attempts"):
            if metric in metrics:
                metrics[metric] *= math.exp(adjustment(specs.get(f"{position}:{metric}"), features))
        revised.append({**row, "metrics": metrics, "baseline": base})
    # Keep the structural RB and WR/TE pools intact. Individual role changes
    # redistribute work; they cannot create additional team targets/carries.
    pools = defaultdict(list)
    for row in revised:
        group = "RB" if row["position"] == "RB" else "WR_TE"
        for metric in ("targets", "carries"):
            if metric in row["metrics"]:
                pools[(row["game_id"], row["team"], group, metric)].append(row)
    for (_, _, _, metric), members in pools.items():
        old = sum(r["baseline"][metric] for r in members)
        new = sum(r["metrics"][metric] for r in members)
        if new > 0:
            for row in members:
                row["metrics"][metric] *= old / new
    by_team = defaultdict(list)
    for row in revised:
        by_team[(row["game_id"], row["team"])].append(row)
    for members in by_team.values():
        quarterback = next((r for r in members if r["position"] == "QB"), None)
        if quarterback is None:
            continue
        old_qb, new_qb = quarterback["baseline"], quarterback["metrics"]
        pass_scale = new_qb.get("pass_attempts", 0) / max(old_qb.get("pass_attempts", 0), 0.1)
        backs = [r for r in members if "carries" in r["metrics"]]
        back_total = sum(r["metrics"]["carries"] for r in backs)
        back_budget = max(
            0.0, back_total + old_qb.get("rush_attempts", 0) - new_qb.get("rush_attempts", 0)
        )
        for row in members:
            if "targets" in row["metrics"]:
                row["metrics"]["targets"] *= pass_scale
        if back_total > 0:
            for row in backs:
                row["metrics"]["carries"] *= back_budget / back_total
    for row in revised:
        base, metrics, features = row["baseline"], row["metrics"], row["features"]
        position = row["position"]
        components = {}
        mappings = [
            ("receiving_yards", "targets", "rate", 0.5, 18.0),
            ("receptions", "targets", "rate", 0.05, 0.98),
            ("passing_yards", "pass_attempts", "rate", 2.0, 13.0),
            ("completions", "pass_attempts", "rate", 0.20, 0.95),
            (
                "rushing_yards",
                "carries" if position == "RB" else "rush_attempts",
                "rate",
                0.5,
                12.0,
            ),
        ]
        for metric, opportunity, _, low, high in mappings:
            if metric not in base or opportunity not in base or base[opportunity] <= 0:
                continue
            old_rate = base[metric] / base[opportunity]
            delta = adjustment(specs.get(f"{position}:{metric}"), features)
            rate = max(low, min(high, old_rate + delta))
            value = rate * metrics[opportunity]
            components[metric] = {
                "structural_mean": round(base[metric], 3),
                "opportunity_effect": round(
                    old_rate * (metrics[opportunity] - base[opportunity]), 3
                ),
                "rate_effect": round((rate - old_rate) * metrics[opportunity], 3),
                "adjusted_rate": round(rate, 4),
                "final_mean": round(value, 3),
            }
            metrics[metric] = value
        for metric in ("passing_tds", "interceptions", "anytime_td_probability"):
            if metric not in base:
                continue
            delta = adjustment(specs.get(f"{position}:{metric}"), features)
            scale = math.exp(delta)
            if metric == "anytime_td_probability":
                before = max(1e-5, min(0.99999, base[metric]))
                old_opps = base.get("targets", 0) + base.get("carries", 0)
                new_opps = metrics.get("targets", 0) + metrics.get("carries", 0)
                volume_scale = new_opps / old_opps if old_opps > 0 else 1.0
                metrics[metric] = 1 - math.exp(math.log1p(-before) * scale * volume_scale)
            else:
                volume_scale = metrics.get("pass_attempts", 1) / max(
                    base.get("pass_attempts", 1), 0.1
                )
                metrics[metric] = base[metric] * scale * volume_scale
        for metric in metrics:
            if metric not in components:
                components[metric] = {
                    "structural_mean": round(base[metric], 3),
                    "adjustment": round(metrics[metric] - base[metric], 3),
                    "final_mean": round(metrics[metric], 3),
                }
        row["components"] = components
        row["metrics"] = {key: round(max(0.0, value), 2) for key, value in metrics.items()}
    return revised


def parameter_status() -> dict:
    params = parameters()
    path = Path(__file__).with_name("prop_parameters.json")
    import hashlib

    return {
        "version": VERSION,
        "state": params.get("state", "not_fitted"),
        "fit_seasons": params.get("fit_seasons", []),
        "selection_season": params.get("selection_season"),
        "held_out_season": params.get("held_out_season"),
        "accepted_corrections": sorted(params.get("corrections", {})),
        "parameter_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        if path.exists()
        else None,
        "market_calibration": "previous projection-version betting calibration does not transfer",
    }


def predictability(metric: str, mean: float, dist: dict | None, evidence: dict) -> dict:
    if dist is None:
        return {"label": "limited evidence", "rank": 3, "range_available": False}
    width = (dist["p90"] - dist["p10"]) / max(mean, 0.25)
    limited = evidence.get("current_stint_games", 0) < 4
    returning = evidence.get("recent_workload_gap_team_games", 0) >= 2
    unresolved = any(
        "availability is unresolved" in message for message in evidence.get("warnings", [])
    )
    volatile = evidence.get("recent_opportunity_cv", 1) > 0.5
    if limited or unresolved or returning:
        label = (
            "availability uncertain"
            if unresolved
            else "recent role unobserved"
            if returning
            else "limited role evidence"
        )
        rank = 3
    elif width <= 1.0 and not volatile:
        label, rank = "relatively stable", 0
    elif width <= 2.0:
        label, rank = "variable", 1
    else:
        label, rank = "high variance", 2
    return {
        "label": label,
        "rank": rank,
        "range_available": True,
        "relative_80_band_width": round(width, 3),
        "p10": dist["p10"],
        "median": dist["p50"],
        "p90": dist["p90"],
        "calibration_n": dist.get("calibration_n"),
        "basis": "Outcome variability and observed role stability; "
        "not probability of a winning bet",
    }


def revise(
    projections: list,
    *,
    history_rows: list[dict],
    pbp_rows: list[dict],
    participation_rows: list[dict],
    charting_rows: list[dict],
    season: int,
    week: int,
    scheme_matchups: dict,
) -> list:
    from dataclasses import replace

    from . import prop_distributions

    params = parameters()
    if params.get("state") != "time_forward_research":
        raise ValueError("Player-intelligence parameters are missing or not validated")
    if params.get("fit_season_max", season) >= season:
        raise ValueError("Player-intelligence parameters are not prior to the forecast season")
    if params.get("model_version") != "nfl-player-projections/3.0.0":
        raise ValueError("Player-intelligence parameters belong to a different projection version")
    histories = history_index(history_rows, season=season, week=week)
    processes = process_games(
        pbp_rows,
        season=season,
        week=week,
        participation_rows=participation_rows,
        charting_rows=charting_rows,
    )
    totals = team_totals(history_rows, season=season, week=week)
    rows = []
    for player in projections:
        features, evidence = feature_profile(
            player,
            histories,
            processes,
            matchup=scheme_matchups.get((player.team, player.opponent)),
            team_totals=totals,
        )
        evidence["current_role_weight"] = round(getattr(player, "current_role_weight", 0), 4)
        rows.append(
            {
                "player_id": player.player_id,
                "game_id": player.game_id,
                "position": player.position,
                "team": player.team,
                "metrics": player.metrics,
                "features": features,
                "evidence": evidence,
            }
        )
    add_quarterback_features(rows, totals, season=season, week=week)
    corrected = correct_metrics(rows, params)
    out = []
    for player, row in zip(projections, corrected):
        evidence = row["evidence"]
        peers = [p for p in projections if p.team == player.team
                 and p.player_id != player.player_id
                 and str(p.injury_status or "").lower() in {"questionable", "doubtful"}
                 and (p.position == "QB" or p.position == player.position
                      or {p.position, player.position} <= {"WR", "TE"})]
        evidence["availability_assumptions"] = [
            {"player": p.player_name, "status": p.injury_status,
             "assumption": "included in the base opportunity allocation"} for p in peers]
        if peers:
            names = ", ".join(p.player_name for p in peers[:3])
            evidence["warnings"].append(f"Teammate availability is unresolved ({names}); "
                                        "the base role assumes those players are available")
        evidence["features"] = {k: round(v, 5) for k, v in row["features"].items()}
        evidence["model_selection"] = "Corrections accepted on 2025; fit ends in the 2025 season"
        development = {}
        for metric, value in row["metrics"].items():
            recent = evidence["recent_metric_means"].get(metric)
            if recent is None or evidence["recent_games"] < 2:
                continue
            threshold = max(5.0 if "yards" in metric else 0.5, 0.10 * recent)
            label = (
                "progression candidate"
                if value - recent > threshold
                else "regression candidate"
                if recent - value > threshold
                else "near recent level"
            )
            development[metric] = {
                "label": label,
                "recent_mean": recent,
                "forecast_mean": value,
                "change": round(value - recent, 3),
                "basis": "Forecast vs last three appearances; role and matchup "
                "effects are separated from efficiency in forecast_components",
            }
        evidence["regression_progression"] = development
        distributions = prop_distributions.stats(
            row["metrics"], position=player.position, evidence=evidence
        )
        predictabilities = {
            metric: predictability(metric, mean, distributions.get(metric), evidence)
            for metric, mean in row["metrics"].items()
        }
        primary = {
            "QB": "pass_attempts",
            "RB": "carries",
            "WR": "targets",
            "TE": "targets",
            "K": "fg_attempts",
        }[player.position]
        ranking = predictabilities.get(primary, {}).get("rank", 3)
        confidence = "high" if ranking == 0 else ("medium" if ranking == 1 else "low")
        out.append(
            replace(
                player,
                metrics=row["metrics"],
                intelligence=evidence,
                predictability=predictabilities,
                forecast_components=row["components"],
                confidence=confidence,
            )
        )
    return out


def quote_comparisons(slate) -> list[dict]:
    """Every matched posted line, without borrowing a different model's edge record."""
    from . import prop_distributions
    from .recommendations import PROP_METRICS
    from .sources.oddsapi import normalise

    projections = getattr(slate, "player_projections", [])
    by_id = {p.player_id: p for p in projections}
    by_name = {normalise(p.player_name): p for p in projections}
    comparisons = []
    for quote in getattr(slate, "player_prop_quotes", []):
        player = (
            by_id.get(quote.player_id)
            if getattr(quote, "player_id", None)
            else by_name.get(normalise(quote.player_name))
        )
        spec = PROP_METRICS.get(quote.market)
        line = number(quote.line)
        if player is None or spec is None or line is None or not math.isfinite(line):
            continue
        metric, label, _ = spec
        if metric == "rush_attempts" and player.position == "RB":
            metric = "carries"
        mean = player.metrics.get(metric)
        if mean is None:
            continue
        distribution = prop_distributions.for_player(player, metric)
        priced = getattr(quote, "priced", True)
        comparisons.append(
            {
                "player_id": player.player_id,
                "player": player.player_name,
                "position": player.position,
                "team": player.team,
                "opponent": player.opponent,
                "metric": metric,
                "market": label,
                "model_mean": mean,
                "model_median": distribution["p50"] if distribution else None,
                "p10": distribution["p10"] if distribution else None,
                "p90": distribution["p90"] if distribution else None,
                "line": line,
                "book": quote.book_title,
                "over_price": quote.over_price if priced else None,
                "under_price": quote.under_price if priced else None,
                "priced": priced,
                "quote_updated_at": quote.last_update,
                "projection_version": player.model_version,
                "predictability": getattr(player, "predictability", {}).get(metric, {}),
                "market_calibrated": False,
                "comparison_status": "posted_line_and_projection; no validated v3 betting edge",
            }
        )
    return sorted(
        comparisons, key=lambda r: (r["predictability"].get("rank", 3), r["player"], r["metric"])
    )
