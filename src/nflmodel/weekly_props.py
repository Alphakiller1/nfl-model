"""Four weekly top tens, ranked by threshold likelihood under football stress.

Use the fitted matrix once, the existing line calibration for supported book
markets, and explicit distribution assumptions for research extensions. Never
rank a fabricated sportsbook line or use a generic team scheme bonus. There
is one entry per player; quoted and research thresholds occupy separate tiers.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from . import prop_distributions, prop_pricing, prop_scouting, teams
from .sources.chase_context import timestamp
from .sources.oddsapi import normalise

VERSION = "nfl-weekly-schematic-props/1.0.0"
GROUPS = ("RB", "QB", "WR", "K")
TOP_N = 10
MARKETS = {
    "player_pass_yds": ("passing_yards", "Passing yards"),
    "player_pass_tds": ("passing_tds", "Passing touchdowns"),
    "player_pass_attempts": ("pass_attempts", "Pass attempts"),
    "player_pass_completions": ("completions", "Completions"),
    "player_pass_interceptions": ("interceptions", "Interceptions"),
    "player_rush_yds": ("rushing_yards", "Rushing yards"),
    "player_rush_attempts": ("rush_attempts", "Rush attempts"),
    "player_reception_yds": ("receiving_yards", "Receiving yards"),
    "player_receptions": ("receptions", "Receptions"),
    "player_rush_reception_yds": ("rush_receiving_yards", "Rush + receiving yards"),
    "player_pass_rush_yds": ("pass_rush_yards", "Pass + rushing yards"),
    "player_field_goals": ("fg_made", "Field goals made"),
    "player_pats": ("pat_made", "Extra points made"),
    "player_kicking_points": ("kicking_points", "Kicking points"),
}
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
# Standard research milestones, not inferred sportsbook prices. Choose a
# meaningful hurdle with raw/bounded probability nearest 65%, not an easy
# arbitrary line that lets every player claim 90% confidence.
MILESTONES = {
    "rushing_yards": (20, 30, 40, 50, 60, 70, 80, 90),
    "receiving_yards": (10, 20, 30, 40, 50, 60, 70, 80, 90),
    "rush_receiving_yards": (40, 50, 60, 70, 80, 90, 100, 110),
    "pass_rush_yards": (175, 200, 225, 250, 275, 300),
    "passing_yards": (150, 175, 200, 225, 250, 275, 300),
    "pass_attempts": (24.5, 27.5, 30.5, 33.5, 36.5),
    "completions": (14.5, 17.5, 20.5, 23.5),
    "carries": (9.5, 12.5, 15.5, 18.5, 21.5),
    "rush_attempts": (2.5, 3.5, 4.5, 5.5),
    "receptions": (1.5, 2.5, 3.5, 4.5, 5.5, 6.5),
    "passing_tds": (0.5, 1.5, 2.5),
    "fg_made": (0.5, 1.5, 2.5),
    "pat_made": (1.5, 2.5, 3.5),
    "kicking_points": (4.5, 5.5, 6.5, 7.5, 8.5, 9.5),
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
            # Research milestones are converted to x-.5 before reaching here.
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


def _row(
    player, metric: str, label: str, line: float, scouting: dict, *, quote=None
) -> dict | None:
    if player.position == "QB" and metric in {"rushing_yards", "rush_attempts"}:
        individual = next(s for s in scouting["sections"]
                          if s["family"] == "individual_look_response")
        if not individual.get("rushing"):
            # RB front response cannot establish designed-QB runs, scrambles or kneels.
            return None
    metrics = _metrics(player)
    base = probability(metric, metrics, line, quoted=quote is not None)
    if base is None:
        return None
    side = ("OVER" if base["over"] >= base["under"] else "UNDER") if quote else "OVER"
    direction = side.lower()
    # The explanation needs derived means but must not mutate the projection.
    from dataclasses import replace

    explained_player = replace(player, metrics=metrics)
    angle = prop_scouting.explain(explained_player, metric, line, side, scouting)
    stress, reasons = _stress(player, scouting, angle)
    scenarios = []
    for scale, name in (
        (1 - stress, "reduced volume/efficiency"),
        (1.0, "base"),
        (1 + stress, "expanded volume/efficiency"),
    ):
        shifted = {key: value * scale for key, value in metrics.items()}
        view = probability(metric, shifted, line, quoted=quote is not None)
        scenarios.append(
            {
                "scenario": name,
                "mean_scale": round(scale, 2),
                "hit_probability": round(view[direction], 6),
                "push_probability": round(view["push"], 6),
            }
        )
    conservative = min(s["hit_probability"] for s in scenarios)
    quoted = quote is not None
    # Assumption-based kicker/combined markets stay research even with a book
    # line. Neither nominal ESPN prices nor a likelihood rank authorizes a bet.
    tier = 0 if quoted and base["calibrated"] else 1 if quoted else 2
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
        "selection": side,
        "threshold": line,
        "line": line if quoted else None,
        "threshold_source": "observed_book_line" if quoted else "research_milestone",
        "model_mean": metrics[metric],
        "hit_probability": round(base[direction], 6),
        "hit_probability_interval": base.get(f"{direction}_interval"),
        "push_probability": round(base["push"], 6),
        "conservative_hit_probability": conservative,
        "probability_basis": base["basis"],
        "probability_assumption": base["assumption"],
        "calibrated": base["calibrated"],
        "rank_tier": tier,
        "stress_fraction": stress,
        "stress_reasons": reasons,
        "scenarios": scenarios,
        "book": quote.book_title if quoted else None,
        "quote_updated_at": quote.last_update if quoted else None,
        "price": (
            (quote.over_price if side == "OVER" else quote.under_price)
            if quoted and quote.priced
            else None
        ),
        "priced": bool(quoted and quote.priced),
        "authority": "RESEARCH_ONLY",
        "may_bet": False,
        "action": "RESEARCH — recheck line, role and inactive lists",
        "status": "quoted_research" if quoted else "research_threshold",
        "external_context_available": scouting["external_context_available"],
        "projection_version": player.model_version,
        **angle,
    }


def _research_rows(player, scouting: dict) -> list[dict]:
    out = []
    metrics = _metrics(player)
    for metric in POSITION_METRICS[player.position]:
        if metric not in MILESTONES or metrics.get(metric, 0) <= 0:
            continue
        candidates = []
        for milestone in MILESTONES[metric]:
            # An integer milestone is an at-least hurdle, encoded at x-.5.
            line = milestone - 0.5 if float(milestone).is_integer() else milestone
            p = probability(metric, metrics, line)
            if p:
                candidates.append((abs(p["over"] - 0.65), line, p["over"]))
        if not candidates:
            continue
        _, line, p = min(candidates)
        # Do not fill the list with an effectively impossible or nearly certain
        # threshold. This is a research watchlist with standard, meaningful hurdles.
        if not 0.45 <= p <= 0.85:
            continue
        label = next(
            (label for key, label in MARKETS.values() if key == metric),
            metric.replace("_", " ").title(),
        )
        row = _row(player, metric, label, line, scouting)
        if row:
            out.append(row)
    return out


def _sort(row: dict) -> tuple:
    return (
        row["rank_tier"],
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
    names = {}
    for player in players.values():
        names.setdefault(normalise(player.player_name), []).append(player)
    candidates = {group: [] for group in GROUPS}
    seen = set()
    for quote in slate.player_prop_quotes:
        spec = MARKETS.get(quote.market)
        if spec is None:
            continue
        player = players.get(quote.player_id) if quote.player_id else None
        if not quote.player_id:
            matches = names.get(normalise(quote.player_name), [])
            player = matches[0] if len(matches) == 1 else None
        if player is None:
            continue
        if {teams.canonical(quote.home_team), teams.canonical(quote.away_team)} != {
            player.team,
            player.opponent,
        }:
            excluded["quote_fixture_mismatch"] += 1
            continue
        updated = timestamp(quote.last_update)
        if updated and (updated > moment or (moment - updated).total_seconds() > 72 * 3600):
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
        metric, label = spec
        if metric == "rush_attempts" and player.position == "RB":
            metric = "carries"
        if metric not in POSITION_METRICS[player.position] or _metrics(player).get(metric, 0) <= 0:
            continue
        identity = (player.player_id, metric, line, quote.book)
        if identity in seen:
            continue
        seen.add(identity)
        row = _row(player, metric, label, line, dossiers[player.player_id], quote=quote)
        if row:
            if updated is None:
                row["failure_paths"].append("Quote timestamp unavailable; verify the current line")
            candidates[player.position].append(row)
    for player in players.values():
        candidates[player.position].extend(_research_rows(player, dossiers[player.player_id]))
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
        counts = Counter(row["threshold_source"] for row in ordered)
        groups[group] = {
            "label": "Kicking" if group == "K" else group,
            "requested": TOP_N,
            "published": len(ordered),
            "rows": ordered,
            "quoted": counts["observed_book_line"],
            "research_thresholds": counts["research_milestone"],
            "candidate_players": len({r["player_id"] for r in candidates[group]}),
            "shortfall": max(TOP_N - len(ordered), 0),
            "availability_note": "Ten distinct players, one prop each"
            if len(ordered) == TOP_N
            else "Fewer than ten eligible pre-kickoff players/meaningful thresholds; "
                 "no fabricated entries",
        }
    return {
        "schema": VERSION,
        "season": slate.season,
        "week": slate.week,
        "generated_at_utc": moment.isoformat(),
        "authority": "RESEARCH_ONLY",
        "may_bet": False,
        "method": "Four top tens, one prop per player. Quoted calibrated lines, then quoted "
        "research markets, then research milestones; within each tier sort by lowest "
        "hit probability across disclosed football stress scenarios. Price/EV is not "
        "the ranking objective. Missing assignments are never scored as observed facts.",
        "context_status": {
            k: v for k, v in (getattr(slate, "prop_context", {}) or {}).items() if k != "games"
        },
        "calibration_evidence": prop_pricing.EVIDENCE,
        "stress_note": "Scenario assumptions are sensitivity tests, not fitted penalties or "
        "statistical confidence intervals. New kicking and combined markets need "
        "a forward calibration record before any betting promotion.",
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
