"""Four weekly top tens, ranked by threshold likelihood under football stress.

Use the fitted matrix once and observed PrizePicks thresholds only. No
sportsbook calibration, substituted lines, or invented research milestones.
Distribution assumptions and source availability remain explicit.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from . import prop_distributions, prop_pricing, prop_scouting
from .sources import prizepicks
from .sources.chase_context import timestamp

VERSION = "nfl-weekly-schematic-props/2.0.0"
GROUPS = ("RB", "QB", "WR", "K")
TOP_N = 10
POSITION_METRICS = {
    "RB": ("rushing_yards", "carries", "receiving_yards", "receptions", "rush_receiving_yards"),
    "QB": (
        "passing_yards",
        "pass_attempts",
        "completions",
        "passing_tds",
        "interceptions",
        "rushing_yards",
        "rush_attempts",
        "pass_rush_yards",
    ),
    "WR": ("receiving_yards", "receptions"),
    "K": ("fg_made", "pat_made", "kicking_points"),
}
COMBINED = {
    "rush_receiving_yards": (("rushing_yards", 1), ("receiving_yards", 1)),
    "pass_rush_yards": (("passing_yards", 1), ("rushing_yards", 1)),
    "kicking_points": (("fg_made", 3), ("pat_made", 1)),
}


def _poisson(mean: float, line: float) -> tuple[float, float, float]:
    """Exact discrete win/lose/push masses (tail included in the over)."""
    if line < 0:
        return 1.0, 0.0, 0.0
    under, push = 0.0, 0.0
    for k in range(math.floor(line) + 1):
        mass = (
            (1.0 if k == 0 else 0.0)
            if mean == 0
            else math.exp(-mean + k * math.log(mean) - math.lgamma(k + 1))
        )
        if k < line:
            under += mass
        else:
            push += mass
    return max(0.0, 1.0 - under - push), under, push


def _marginal(metric: str, mean: float, line: float) -> tuple[float, float, float] | None:
    if metric in {"fg_made", "pat_made"}:
        return _poisson(mean, line)
    if line < 0:
        return 1.0, 0.0, 0.0
    if mean <= 0:
        return (0.0, 1.0, 0.0) if line > 0 else (0.0, 0.0, 1.0)
    dist = prop_distributions.distribution(metric, mean)
    if dist is None:
        return None
    if dist.get("pmf"):
        pmf = {int(k): v for k, v in dist["pmf"].items()}
        total = sum(pmf.values())
        over = sum(v for k, v in pmf.items() if k > line) / total
        under = sum(v for k, v in pmf.items() if k < line) / total
    else:
        # The empirical ladder is continuous; integer outcomes are approximated
        # with half-unit boundaries so a push never silently becomes an under.
        over = prop_distributions.over_probability(dist, math.floor(line) + 0.5)
        under = 1 - prop_distributions.over_probability(dist, math.ceil(line) - 0.5)
    return over, under, max(0.0, 1.0 - over - under)


def probability(metric: str, metrics: dict, line: float, *, quoted: bool = False) -> dict | None:
    """Win probabilities or dependence bounds, with pushes explicitly separated.

    Combined-yardage/kicking-point probabilities use Frechet/union bounds on
    the marginals: max(P(A>a)+P(B>b)-1) <= P(A+B>t) <=
    min(P(A>a)+P(B>b)), a+b=t. No independence/correlation fit is invented.
    Integer combined lines are excluded because their push mass is unmeasured.
    """
    if metric in COMBINED:
        if float(line).is_integer():
            # Combined integer lines have no validated tie mass.
            return None
        (left, ls), (right, rs) = COMBINED[metric]
        if any(key not in metrics for key in (left, right)):
            return None
        lower, upper = 0.0, 1.0
        # All possible integer cut points: lattice outcomes, no random sampling.
        for cut in range(-1, math.ceil(line) + 1):
            a = _marginal(left, metrics[left], cut / ls)
            b = _marginal(right, metrics[right], (line - cut) / rs)
            if a is None or b is None:
                return None
            lower = max(lower, a[0] + b[0] - 1)
            upper = min(upper, a[0] + b[0])
        lower, upper = min(max(lower, 0.0), 1.0), min(max(upper, 0.0), 1.0)
        upper = max(lower, upper)
        return {
            "over": lower,
            "under": 1 - upper,
            "push": 0.0,
            "over_interval": [lower, upper],
            "under_interval": [1 - upper, 1 - lower],
            "basis": "dependence_bound",
            "calibrated": False,
            "assumption": "Marginal outcome models; unknown dependence is bounded. "
            "Kicking marginals assume Poisson counts."
            if metric == "kicking_points"
            else "Fitted marginal models; unknown dependence is bounded",
        }
    mean = metrics.get(metric)
    if mean is None or not math.isfinite(mean) or mean < 0:
        return None
    masses = _marginal(metric, mean, line)
    if masses is None:
        return None
    over, under, push = masses
    calibrated = quoted and metric in prop_pricing.FAMILY
    if calibrated and 1 - push > 1e-9:
        conditional = prop_pricing.calibrate(over / (1 - push))
        over, under = conditional * (1 - push), (1 - conditional) * (1 - push)
    return {
        "over": over,
        "under": under,
        "push": push,
        "basis": "line_calibrated"
        if calibrated
        else ("poisson_assumption" if metric in {"fg_made", "pat_made"} else "raw_distribution"),
        "calibrated": calibrated,
        "assumption": "Existing projection-error distribution and line calibration"
        if calibrated
        else "Poisson count assumption; unvalidated for kicker markets"
        if metric in {"fg_made", "pat_made"}
        else "Fitted projection-error distribution; no line calibration",
    }


def _metrics(player) -> dict:
    out = dict(player.metrics)
    if player.position == "RB" and "carries" in out:
        out["rush_attempts"] = out["carries"]
    for key, components in COMBINED.items():
        if all(metric in out for metric, _ in components):
            out[key] = round(sum(out[metric] * scale for metric, scale in components), 2)
    return out


def _stress(player, scouting: dict, angle: dict) -> tuple[float, list[str]]:
    # Scenario assumptions, not learned probability penalties or confidence
    # intervals. Rank on the least favorable mean in this disclosed envelope.
    fraction = 0.15
    reasons = ["Base ±15% opportunity/efficiency stress"]
    if player.position == "K":
        fraction = 0.25
        reasons = ["Kicker ±25% opportunity/make-rate stress"]
    if player.history_games < 4 or player.role_continuity != "same team":
        fraction += 0.05
        reasons.append("Limited same-role history adds 5 percentage points of stress")
    if player.injury_status:
        fraction += 0.10
        reasons.append("Player injury designation adds 10 percentage points of stress")
    if not scouting["external_context_available"]:
        fraction += 0.05
        reasons.append("Missing dated player/trench evidence adds 5 percentage points of stress")
    if angle["conflicts"]:
        fraction += 0.05
        reasons.append("Opposing schematic evidence adds 5 percentage points of stress")
    return min(fraction, 0.40), reasons


def _row(player, metric: str, label: str, line: float, scouting: dict, *, quote) -> dict | None:
    if player.position == "QB" and metric in {"rushing_yards", "rush_attempts"}:
        individual = next(
            s for s in scouting["sections"] if s["family"] == "individual_look_response"
        )
        if not individual.get("rushing"):
            return None
    metrics = _metrics(player)
    # The DraftKings line calibration has no PrizePicks validation record.
    base = probability(metric, metrics, line, quoted=False)
    if base is None:
        return None
    offered = {"MORE": "over", "LESS": "under"}
    choices = [side for side in quote.published_sides if side in offered]
    if not choices:
        return None
    selection = max(choices, key=lambda side: base[offered[side]])
    direction = offered[selection]
    from dataclasses import replace

    explained_player = replace(player, metrics=metrics)
    angle = prop_scouting.explain(
        explained_player, metric, line, "OVER" if selection == "MORE" else "UNDER", scouting
    )
    if not quote.entry_availability_verified:
        angle["failure_paths"].append(
            "Public PrizePicks card: confirm standard/Goblin/Demon status, offered side "
            "and current line in the app; contest availability is unverified"
        )
    stress, reasons = _stress(player, scouting, angle)
    scenarios = []
    for scale, name in (
        (1 - stress, "reduced volume/efficiency"),
        (1.0, "base"),
        (1 + stress, "expanded volume/efficiency"),
    ):
        shifted = {key: value * scale for key, value in metrics.items()}
        view = probability(metric, shifted, line, quoted=False)
        scenarios.append(
            {
                "scenario": name,
                "mean_scale": round(scale, 2),
                "hit_probability": round(view[direction], 6),
                "push_probability": round(view["push"], 6),
            }
        )
    return {
        "player": player.player_name,
        "player_id": player.player_id,
        "position": player.position,
        "team": player.team,
        "opponent": player.opponent,
        "season": player.season,
        "week": player.week,
        "game_id": player.game_id,
        "kickoff_utc": player.kickoff_utc,
        "metric": metric,
        "market": label,
        "selection": selection,
        "threshold": line,
        "line": line,
        "threshold_source": "prizepicks_published_projection",
        "line_provider": "PrizePicks",
        "provider_projection_id": quote.projection_id,
        "source_stat": quote.source_stat,
        "source_url": quote.source_url,
        "line_variant": quote.variant,
        "published_sides": list(quote.published_sides),
        "entry_availability_verified": quote.entry_availability_verified,
        "model_mean": metrics[metric],
        "hit_probability": round(base[direction], 6),
        "hit_probability_interval": base.get(f"{direction}_interval"),
        "push_probability": round(base["push"], 6),
        "conservative_hit_probability": min(s["hit_probability"] for s in scenarios),
        "probability_basis": base["basis"],
        "probability_assumption": base["assumption"],
        "calibrated": False,
        "stress_fraction": stress,
        "stress_reasons": reasons,
        "scenarios": scenarios,
        "book": "PrizePicks",
        "quote_updated_at": quote.observed_at_utc,
        "price": None,
        "priced": False,
        "authority": "RESEARCH_ONLY",
        "may_bet": False,
        "action": "RESEARCH — recheck PrizePicks line, variant, offered side and inactive lists",
        "status": "posted_prizepicks_research",
        "external_context_available": scouting["external_context_available"],
        "projection_version": player.model_version,
        **angle,
    }


def _sort(row: dict) -> tuple:
    return (
        -row["conservative_hit_probability"],
        -row["hit_probability"],
        len(row["conflicts"]),
        row["player_id"],
        row["metric"],
    )


def build(slate, *, now: datetime | None = None) -> dict:
    moment = now or timestamp(getattr(slate, "assembled_at_utc", None)) or datetime.now(UTC)
    excluded = Counter()
    players, dossiers = {}, {}
    for player in slate.player_projections:
        if (
            player.position not in GROUPS
            or player.season != slate.season
            or player.week != slate.week
        ):
            continue
        kickoff = timestamp(player.kickoff_utc)
        if kickoff is None or kickoff <= moment:
            excluded["started_or_undated_game"] += 1
            continue
        if player.depth_rank > {"QB": 1, "K": 1, "RB": 2, "WR": 3}[player.position]:
            excluded["outside_material_depth_role"] += 1
            continue
        scouting = prop_scouting.dossier(slate, player)
        if prop_scouting.player_unavailable(player, scouting):
            excluded["unavailable_player"] += 1
            continue
        if not player.player_id or not all(
            isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
            for v in player.metrics.values()
        ):
            excluded["invalid_projection"] += 1
            continue
        players[player.player_id] = player
        dossiers[player.player_id] = scouting
    candidates = {group: [] for group in GROUPS}
    seen = set()
    # Dedicated PrizePicks list: generic sportsbook quotes cannot reach this path.
    for quote in getattr(slate, "prizepicks_quotes", []):
        if not isinstance(quote, prizepicks.PrizePicksLine) or quote.provider != "prizepicks":
            excluded["non_prizepicks_source"] += 1
            continue
        player = players.get(quote.player_id)
        if player is None:
            continue
        kickoff, observed = timestamp(quote.kickoff_utc), timestamp(quote.observed_at_utc)
        if (
            quote.team != player.team
            or quote.opponent != player.opponent
            or kickoff is None
            or (abs((kickoff - timestamp(player.kickoff_utc)).total_seconds()) >= 60)
        ):
            excluded["quote_fixture_mismatch"] += 1
            continue
        if observed is None or not 0 <= (moment - observed).total_seconds() <= (
            prizepicks.MAX_AGE_SECONDS
        ):
            excluded["stale_or_future_quote"] += 1
            continue
        try:
            line = float(quote.line)
        except (ValueError, TypeError):
            excluded["invalid_line"] += 1
            continue
        if not math.isfinite(line) or line < 0:
            excluded["invalid_line"] += 1
            continue
        spec = prizepicks.MARKETS.get(quote.source_stat)
        if (
            spec != (quote.metric, quote.market)
            or not isinstance(quote.projection_id, str)
            or not quote.projection_id
            or not isinstance(quote.published_sides, (tuple, list))
            or not isinstance(quote.source_url, str)
            or not quote.source_url.startswith(prizepicks.BASE)
        ):
            excluded["unsupported_or_unattributed_projection"] += 1
            continue
        metric = (
            "carries"
            if quote.metric == "rush_attempts" and player.position == "RB"
            else (quote.metric)
        )
        if metric not in POSITION_METRICS[player.position] or _metrics(player).get(metric, 0) <= 0:
            continue
        identity = quote.projection_id
        if identity in seen:
            continue
        seen.add(identity)
        row = _row(player, metric, quote.market, line, dossiers[player.player_id], quote=quote)
        if row:
            candidates[player.position].append(row)
    groups = {}
    for group in GROUPS:
        ordered, used = [], set()
        for row in sorted(candidates[group], key=_sort):
            if row["player_id"] in used:
                continue
            used.add(row["player_id"])
            ordered.append({"rank": len(ordered) + 1, **row})
            if len(ordered) == TOP_N:
                break
        groups[group] = {
            "label": "Kicking" if group == "K" else group,
            "requested": TOP_N,
            "published": len(ordered),
            "rows": ordered,
            "quoted": len(ordered),
            "research_thresholds": 0,
            "candidate_players": len({r["player_id"] for r in candidates[group]}),
            "shortfall": max(TOP_N - len(ordered), 0),
            "availability_note": "Ten distinct players, one prop each"
            if len(ordered) == TOP_N
            else "Fewer than ten eligible players with observed PrizePicks lines; "
            "no fabricated entries",
        }
    return {
        "schema": VERSION,
        "season": slate.season,
        "week": slate.week,
        "generated_at_utc": moment.isoformat(),
        "authority": "RESEARCH_ONLY",
        "may_bet": False,
        "line_provider": "PrizePicks",
        "line_source_status": getattr(slate, "prizepicks_status", {}) or {},
        "method": "Four top tens, one actual PrizePicks projection per player. Sort by the "
        "lowest hit probability across disclosed football stress scenarios. No sportsbook "
        "or invented-milestone fallback; no PrizePicks calibration or payout/EV claim.",
        "context_status": {
            k: v for k, v in (getattr(slate, "prop_context", {}) or {}).items() if k != "games"
        },
        "calibration_evidence": {
            "provider": "PrizePicks",
            "status": "unvalidated",
            "note": "DraftKings closing-line calibration is not applied to PrizePicks",
        },
        "stress_note": "Scenario assumptions are sensitivity tests, not fitted penalties or "
        "statistical confidence intervals. New kicking and combined markets need "
        "a forward calibration record before any betting promotion. Public research cards "
        "may include alternate lines; variant and contest availability are unverified.",
        "groups": groups,
        "excluded": dict(excluded),
    }


def for_slate(slate) -> dict:
    """One report shared by HTML, board JSON and the CLI at one assembly time."""
    report = getattr(slate, "weekly_prop_report", None)
    if not report:
        report = build(slate)
        slate.weekly_prop_report = report
    return report


def write(report: dict, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
