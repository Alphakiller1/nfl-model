"""Descriptive run-point and scoring-range matchup paths, never fitted inputs."""

from __future__ import annotations

from . import teams


def run_points(rushing: dict, lanes: dict, baseline: dict) -> dict:
    all_runs = rushing.get("all") or {}
    total, all_yield = all_runs.get("carries", 0), baseline.get("yards_per_carry")
    if not total or all_yield is None:
        return {"status": "unavailable"}
    looks = ("gap_guard", "gap_tackle", "gap_end")
    if not any(key in rushing for key in looks):
        looks = ("left", "middle", "right")
    contributions, effect, mass = [], 0.0, 0.0
    for look in looks:
        split = rushing.get(look) or {}
        lane = lanes.get(look if look.startswith("gap_") else f"lane_{look}") or {}
        n, ypc = lane.get("carries", 0), lane.get("yards_per_carry_allowed")
        if split.get("source_season") != all_runs.get("source_season") or n < 5 or ypc is None:
            continue
        share = split.get("carries", 0) / total
        delta = n / (n + 24) * (ypc - all_yield)
        effect += share * delta
        mass += share
        contributions.append(
            {
                "run_point_proxy": look,
                "player_carry_share": round(share, 3),
                "player_carries": split.get("carries"),
                "opponent_carries": n,
                "opponent_yards_per_carry": ypc,
                "shrunk_delta": round(delta, 3),
            }
        )
    if mass > 1:
        effect /= mass
    return {
        "status": "proxy" if contributions else "unavailable",
        "opponent_baseline_yards_per_carry": all_yield,
        "matched_run_point_yards_per_carry": round(all_yield + effect, 3),
        "response_delta": round(effect, 3),
        "covered_mass": round(min(mass, 1), 3),
        "contributions": contributions,
        "used_to_change_projection": False,
        "mechanism": "Opponent run-point yield weighted by same-season player carry mix; "
        "each lane shrunk to opponent overall yield. Run points do not "
        "identify blocking concepts.",
    }


def _unit(game: dict, team: str, phase: str, window: str = "current") -> dict:
    side = "home" if teams.canonical(game.get("home", "")) == team else "away"
    value = game.get(f"{side}_red_zone") or {}
    return (value.get(window) or {}).get(phase) or (value.get("combined") or {}).get(phase) or {}


def kicking(slate, game: dict, player) -> dict:
    own, opp = _unit(game, player.team, "offense"), _unit(game, player.opponent, "defense")
    if not own or not opp:
        return {"status": "unavailable"}
    pool = {}
    for fixture in (getattr(slate, "prop_context", {}) or {}).get("games", []):
        for side in ("home", "away"):
            club = teams.canonical(fixture.get(side, ""))
            rz = _unit(fixture, club, "offense")
            if club and rz and rz.get("trips", 0) > 0:
                pool[club] = rz
    if len(pool) < 8:
        return {"status": "unavailable", "reason": "Insufficient league drive evidence"}
    trips = sum(r["trips"] for r in pool.values())
    games = sum(r.get("games", 0) for r in pool.values())
    if not trips or not games:
        return {"status": "unavailable"}
    league = {
        "td_rate": sum(r.get("td_trips", 0) for r in pool.values()) / trips,
        "score_rate": sum(r.get("score_trips", 0) for r in pool.values()) / trips,
        "trips_per_game": trips / games,
    }
    matchup = {}
    for key in league:
        values = []
        for club, phase, unit in ((player.team, "offense", own), (player.opponent, "defense", opp)):
            baseline = _unit(game, club, phase, "combined")
            if unit.get(key) is None:
                return {"status": "unavailable"}
            n = unit.get("games", 0) if key == "trips_per_game" else unit.get("trips", 0)
            pseudo = 6 if key == "trips_per_game" else 16
            prior = baseline.get(key, league[key])
            values.append((unit[key] * n + prior * pseudo) / (n + pseudo))
        matchup[key] = sum(values) / 2
    return {
        "status": "proxy",
        "matchup": matchup,
        "league": league,
        "league_clubs": len(pool),
        "offense_trips": own.get("trips"),
        "opponent_trips": opp.get("trips"),
        "used_to_change_projection": False,
        "mechanism": "Offense/opponent scoring-range rates with 16-trip and 6-game "
        "combined-window shrinkage. Non-TD scoring trips proxy kick chances; "
        "outside-red-zone kicks and fourth-down choices are unmeasured.",
    }


def kicking_signals(metric: str, over: bool, path: dict) -> tuple[list, list, str]:
    if path.get("status") != "proxy":
        return [], [], ""
    expected, league = path["matchup"], path["league"]
    td_delta = expected["td_rate"] - league["td_rate"]
    support, conflicts = [], []
    note = (
        f"Shrunk scoring-range TD conversion {expected['td_rate']:.0%} vs "
        f"{league['td_rate']:.0%} league; access {expected['trips_per_game']:.1f} "
        f"trips/game vs {league['trips_per_game']:.1f}. These are drive-path proxies."
    )
    if metric == "pat_made" and abs(td_delta) > 0.04:
        (support if (td_delta > 0) == over else conflicts).append(
            "TD conversion supports "
            + ("more" if td_delta > 0 else "fewer")
            + " PAT opportunities at comparable scoring-range access"
        )
    if metric == "fg_made":
        if td_delta > 0.04:
            (conflicts if over else support).append(
                "Higher TD conversion removes FG chances at comparable drive access"
            )
        elif td_delta < -0.04 and expected["score_rate"] >= league["score_rate"]:
            (support if over else conflicts).append(
                "Lower TD conversion with retained scoring conversion supports stalled-drive kicks"
            )
    if expected["trips_per_game"] < league["trips_per_game"] * 0.90:
        (conflicts if over else support).append(
            "Below-league scoring-range access constrains kick opportunities"
        )
    return support, conflicts, note
