"""Fail a build whose board contradicts itself.

Run after `verify_build.py` on the built site's board.json. Each check is a
mistake the board has actually made (or could make the same way):

* a spread/total pick priced above its market's measured skill (GB @ TB week 4
  2026: "Over 38.5, 76%" from a total model with no skill against the close);
* a pick whose side disagrees with the model number it cites;
* a game whose scoreline does not add up to its total and margin, or whose
  published win probability favours the other side of the published margin;
* a game with a starting quarterback out but no availability adjustment on the
  margin or the total (the total used to ignore it);
* a projected player who is listed Out; a team whose projected targets exceed
  its quarterback's attempts; a negative projection;
* a prop slip leg outside the slip rules, or two legs from one game in a slip;
* a weekly-report gap presented as a call, a gap row that does not name a signal
  pointing the other way, or a prop row calling the side best bets fade.

    PYTHONPATH=src python scripts/audit_consistency.py _site/board.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nflmodel import calibration, player_props, prop_pricing, slips  # noqa: E402

TOL = 0.11   # scores publish to one decimal, so a sum can be off by up to 0.1


def _picks(board: dict) -> list[str]:
    out = []
    players = {p.get("player_id"): p for p in board.get("player_projections") or []}
    for p in board.get("best_bets") or []:
        family, prob = p.get("family"), float(p.get("probability") or 0)
        name = f"pick {p.get('selection')} ({family})"
        if family in calibration.SKILL:
            if prob > calibration.ceiling(family, float(p.get("edge") or 0)):
                out.append(f"{name}: probability {prob:.3f} exceeds the market's measured skill "
                           f"({calibration.probability(family, float(p.get('edge') or 0)):.3f})")
            if prob < calibration.MIN_PUBLISH_PROBABILITY:
                out.append(f"{name}: published below {calibration.MIN_PUBLISH_PROBABILITY}")
        if family == "total":
            over = float(p["model_number"]) > float(p["book_number"])
            if over != (p.get("side") == "over"):
                out.append(f"{name}: side contradicts model {p['model_number']} "
                           f"vs {p['book_number']}")
        if family == "prop":
            top = prop_pricing.calibrate(1.0)
            if prob > top + 1e-6 or prob < 0.5:
                out.append(f"{name}: probability {prob:.3f} outside the calibrated range")
            player = players.get(p.get("player_id")) or {}
            metric = p.get("metric") or ""
            if metric == "rush_attempts" and player.get("position") == "RB":
                metric = "carries"
            if player and not slips.market_allowed(p.get("side"), player.get("position"), metric):
                out.append(f"{name}: market without a record")
            if player and player.get("position") in ("WR", "TE", "RB") and (
                    int(player.get("depth_rank") or 1) >= 3):
                out.append(f"{name}: backup (depth 3+)")
            if float(p.get("line") or 0) < slips.MIN_LINE:
                out.append(f"{name}: 0.5 line")
    return out


def _games(board: dict) -> list[str]:
    out = []
    for g in board.get("games") or []:
        name = f"{g.get('away')} @ {g.get('home')}"
        home, away = g.get("projected_home_score"), g.get("projected_away_score")
        total, margin = g.get("projected_total"), g.get("model_margin")
        if None not in (home, away, total) and min(home, away) > 0:
            if abs(home + away - total) > TOL:
                out.append(f"{name}: scoreline {away}-{home} does not add to total {total}")
            if margin is not None and abs(home - away - margin) > TOL:
                out.append(f"{name}: scoreline {away}-{home} does not match margin {margin}")
        published, win = g.get("published_margin"), g.get("win_probability")
        if published not in (None, 0) and win is not None and (published > 0) != (win > 0.5):
            out.append(f"{name}: win probability {win} favours the other side of {published}")
        if g.get("qb_out"):
            if not g.get("availability_margin") and g.get("availability_margin") != 0.0:
                out.append(f"{name}: QB out but no margin adjustment")
            if g.get("total_modelled") and not (g.get("availability_total") or 0) < 0:
                out.append(f"{name}: QB out ({', '.join(g['qb_out'])}) but the total is unadjusted")
    return out


def _players(board: dict) -> list[str]:
    out = []
    targets = defaultdict(float)
    attempts = {}
    for p in board.get("player_projections") or []:
        name = f"{p.get('player_name')} ({p.get('team')})"
        if str(p.get("injury_status") or "").lower() == "out":
            out.append(f"{name}: projected while listed Out")
        metrics = p.get("metrics") or {}
        if any(float(v) < 0 for v in metrics.values()):
            out.append(f"{name}: negative projection")
        targets[p.get("team")] += float(metrics.get("targets") or 0)
        if p.get("position") == "QB" and int(p.get("depth_rank") or 1) == 1:
            attempts[p.get("team")] = float(metrics.get("pass_attempts") or 0)
    for team, total in targets.items():
        qb = attempts.get(team)
        if qb and total > qb / player_props.QB_ATTEMPT_SHARE + 0.5:
            out.append(f"{team}: projected targets {total:.1f} exceed team attempts "
                       f"{qb / player_props.QB_ATTEMPT_SHARE:.1f}")
    return out


def _slips(board: dict) -> list[str]:
    plan = board.get("prop_slips") or {}
    out, players = [], set()
    for i, slip in enumerate(plan.get("slips") or [], start=1):
        games = [leg["game_id"] for leg in slip["legs"]]
        if len(games) != len(set(games)):
            out.append(f"slip {i}: two legs from one game")
        for leg in slip["legs"]:
            label = f"slip {i} {leg.get('label')}"
            if leg["player_id"] in players:
                out.append(f"{label}: player used twice")
            players.add(leg["player_id"])
            if float(leg["probability"]) < slips.MIN_LEG_PROBABILITY:
                out.append(f"{label}: below the leg threshold")
            if not slips.market_allowed(leg["side"], leg["position"], leg["metric"]):
                out.append(f"{label}: market without a record")
            if float(leg["line"]) < slips.MIN_LINE:
                out.append(f"{label}: 0.5 line")
    return out


def _report(board: dict) -> list[str]:
    """The weekly report must not call what calibration says it cannot, and must name
    every signal that disagrees with a gap (KC @ LV: 'UNDER' beside a matchup
    context that pointed to more offense, with nothing saying they disagreed)."""
    report = board.get("weekly_report") or {}
    out = []
    for key, market in (("top_spreads", "spread"), ("top_totals", "total")):
        for row in report.get(key) or []:
            name = f"report {key} {row.get('game')}"
            words = str(row.get("selection") or "").upper().split()
            if calibration.SKILL[market].weight <= 0.05 and (
                    {"OVER", "UNDER"} & set(words) or len(words) == 1):
                out.append(f"{name}: presented as a call ({row.get('selection')})")
            if "conflicts" not in row:
                out.append(f"{name}: no reconciliation of the signals")
                continue
            season = row.get("season_only")
            if key == "top_totals" and season is not None:
                model, market_total = float(row["model"]), float(row["market"])
                disagrees = ((model - market_total) * (float(season) - market_total) < 0
                             and abs(float(season) - market_total) >= 0.5)
                if disagrees and "this season's form" not in row["conflicts"]:
                    out.append(f"{name}: 2026-only total {season} disagrees but is not named")
    picks = {(p.get("player_id"), p.get("metric")): p.get("side")
             for p in board.get("best_bets") or [] if p.get("family") == "prop"}
    names = {p.get("player_name"): p.get("player_id")
             for p in board.get("player_projections") or []}
    for row in report.get("top_player_props") or []:
        side = picks.get((names.get(row.get("player")), _metric_key(row.get("market_key"))))
        if side and side != str(row.get("selection")).lower():
            out.append(f"report prop {row.get('player')} {row.get('market')}: lists "
                       f"{row.get('selection')} while best bets play {side}")
    return out


def _metric_key(market_key: str | None) -> str | None:
    from nflmodel.best_bets import PROP_MARKETS

    spec = PROP_MARKETS.get(market_key or "")
    return spec[0] if spec else None


def audit(board: dict) -> list[str]:
    return (_picks(board) + _games(board) + _players(board) + _slips(board)
            + _report(board))


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "_site/board.json")
    failures = audit(json.loads(path.read_text(encoding="utf-8")))
    for failure in failures:
        print(f"[FAIL] {failure}")
    print(f"consistency audit: {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
