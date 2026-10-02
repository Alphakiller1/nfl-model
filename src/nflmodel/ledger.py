"""Immutable pre-kickoff NFL projection ledger and deterministic grader.

Every production build records the exact independent projection and DraftKings
quote it displayed.  A later build grades completed games.  This remains a
shadow record: no stored disagreement is retroactively converted into a bet.
"""

from __future__ import annotations

import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from . import teams

if TYPE_CHECKING:
    from .forecast import GameProjection
    from .player_props import PlayerProjection


SCHEMA_VERSION = "1.2.0"

# Ledgers written before 1.2.0 graded player snapshots against stat rows filtered
# to weeks *before* the board week, so a Monday build graded every Sunday player
# as a zero-stat DNP. The projections themselves were stored intact, so those
# rows are re-opened and graded again from the published stat file.
_PLAYER_REGRADE_BEFORE = (1, 2, 0)
DEFAULT_PATH = Path(
    os.getenv(
        "NFL_LEDGER_PATH",
        str(
            Path(__file__).resolve().parents[2]
            / "data" / "runtime-cache" / "prediction-ledger.json"
        ),
    )
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(moment: datetime | None = None) -> str:
    return (moment or _now()).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _version(value) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in str(value).split("."))
    except ValueError:
        return (0,)


def _reopen_player_grades(payload: dict) -> None:
    if _version(payload.get("schema_version")) >= _PLAYER_REGRADE_BEFORE:
        return
    for row in payload.get("player_snapshots", []):
        if row.get("status") != "graded":
            continue
        for key in ("graded_at", "actual_metrics", "absolute_errors", "anytime_td_brier"):
            row.pop(key, None)
        row["status"] = "pending"


def _load(path: Path) -> dict:
    if not path.is_file():
        return {"schema_version": SCHEMA_VERSION, "snapshots": [], "player_snapshots": [],
                "best_bets": [], "sharp_spots": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload.get("snapshots"), list):
            payload.setdefault("player_snapshots", [])
            payload.setdefault("best_bets", [])
            payload.setdefault("sharp_spots", [])
            _reopen_player_grades(payload)
            return payload
    except (json.JSONDecodeError, AttributeError):
        pass
    return {"schema_version": SCHEMA_VERSION, "snapshots": [], "player_snapshots": [],
                "best_bets": [], "sharp_spots": []}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _result_index(games: list[dict]) -> dict[tuple[int, int, str, str], dict]:
    out = {}
    for game in games:
        home_score, away_score = game.get("home_score"), game.get("away_score")
        if home_score is None or away_score is None:
            continue
        key = (
            int(game.get("season") or 0),
            int(game.get("week") or 0),
            str(game.get("home_team") or ""),
            str(game.get("away_team") or ""),
        )
        out[key] = game
    return out


def _grade(snapshot: dict, result: dict) -> None:
    actual_margin = float(result["home_score"]) - float(result["away_score"])
    actual_total = float(result["home_score"]) + float(result["away_score"])
    model_margin = snapshot.get("model_margin")
    consensus_margin = snapshot.get("consensus_margin")
    book_margin = snapshot.get("book_margin")
    model_total = snapshot.get("model_total")
    book_total = snapshot.get("book_total")
    snapshot.update({
        "status": "graded",
        "graded_at": _stamp(),
        "actual_margin": actual_margin,
        "actual_total": actual_total,
        "model_abs_error": (
            abs(float(model_margin) - actual_margin) if model_margin is not None else None
        ),
        "consensus_abs_error": (
            abs(float(consensus_margin) - actual_margin) if consensus_margin is not None else None
        ),
        "book_abs_error": (
            abs(float(book_margin) - actual_margin) if book_margin is not None else None
        ),
        "model_total_abs_error": (
            abs(float(model_total) - actual_total) if model_total is not None else None
        ),
        "book_total_abs_error": (
            abs(float(book_total) - actual_total) if book_total is not None else None
        ),
    })
    if model_margin is not None and book_margin is not None:
        direction = 1 if float(model_margin) > float(book_margin) else -1
        result_edge = actual_margin - float(book_margin)
        snapshot["ats_result"] = (
            "push" if abs(result_edge) < 1e-9 else "win" if result_edge * direction > 0 else "loss"
        )
    if model_total is not None and book_total is not None:
        direction = 1 if float(model_total) > float(book_total) else -1
        result_edge = actual_total - float(book_total)
        snapshot["total_result"] = (
            "push" if abs(result_edge) < 1e-9 else "win" if result_edge * direction > 0 else "loss"
        )


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return round(statistics.fmean(values), 4) if values else None


def _published_team_weeks(rows: list[dict]) -> set[tuple[int, int, str]]:
    """Team-weeks whose stat lines exist in the source at all.

    A missing player row only means "did not play" once the team's box score has
    been published. Before that it means nothing, and grading would record zeros.
    """
    out = set()
    for row in rows:
        team = teams.canonical(row.get("team") or "")
        if team:
            out.add((int(row.get("season") or 0), int(row.get("week") or 0), team))
    return out


def _player_result_index(rows: list[dict]) -> dict[tuple[int, int, str, str], dict]:
    out = {}
    for row in rows:
        player_id = str(row.get("player_id") or "")
        team = teams.canonical(row.get("team") or "")
        if not player_id or not team:
            continue
        key = (
            int(row.get("season") or 0),
            int(row.get("week") or 0),
            team,
            player_id,
        )
        out[key] = row
    return out


def _num(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _actual_player_metrics(row: dict | None, fields: set[str]) -> dict[str, float]:
    source = row or {}
    mapping = {
        "pass_attempts": "attempts",
        "completions": "completions",
        "passing_yards": "passing_yards",
        "passing_tds": "passing_tds",
        "interceptions": "passing_interceptions",
        "rush_attempts": "carries",
        "carries": "carries",
        "rushing_yards": "rushing_yards",
        "targets": "targets",
        "receptions": "receptions",
        "receiving_yards": "receiving_yards",
        "fg_attempts": "fg_att",
        "fg_made": "fg_made",
        "pat_made": "pat_made",
    }
    actual = {field: _num(source, source_field) for field, source_field in mapping.items()
              if field in fields}
    if "anytime_td_probability" in fields:
        actual["anytime_td_probability"] = float(
            _num(source, "rushing_tds") + _num(source, "receiving_tds") > 0
        )
    if "kicking_points" in fields:
        actual["kicking_points"] = 3.0 * _num(source, "fg_made") + _num(
            source, "pat_made"
        )
    return actual


def _grade_player(snapshot: dict, result: dict | None) -> None:
    projected = snapshot.get("metrics") or {}
    actual = _actual_player_metrics(result, set(projected))
    errors = {
        key: abs(float(value) - actual[key])
        for key, value in projected.items()
        if key in actual
    }
    snapshot.update({
        "status": "graded",
        "graded_at": _stamp(),
        "actual_metrics": actual,
        "absolute_errors": errors,
    })
    if "anytime_td_probability" in projected:
        snapshot["anytime_td_brier"] = (
            float(projected["anytime_td_probability"])
            - actual["anytime_td_probability"]
        ) ** 2


def _player_key(row: dict) -> tuple:
    return (row.get("season"), row.get("week"), row.get("game_id"), row.get("player_id"))


def _latest_per_player_game(rows: list[dict]) -> list[dict]:
    """Keep one row per player-game: the latest recorded, graded or not.

    Compacts ledgers written before projections replaced each other in place.
    Order is preserved so the oldest seasons are still the first to be trimmed.
    """
    latest: dict[tuple, int] = {}
    for i, row in enumerate(rows):
        key = _player_key(row)
        held = latest.get(key)
        if held is None or row.get("recorded_at", "") >= rows[held].get("recorded_at", ""):
            latest[key] = i
    keep = set(latest.values())
    return [row for i, row in enumerate(rows) if i in keep]


def _player_summary(payload: dict, season: int) -> dict:
    graded = [
        row for row in payload.get("player_snapshots", [])
        if row.get("status") == "graded" and int(row.get("season", 0)) == season
    ]
    latest: dict[tuple, dict] = {}
    for row in graded:
        key = (row["season"], row["week"], row["game_id"], row["player_id"])
        if key not in latest or row.get("recorded_at", "") > latest[key].get(
            "recorded_at", ""
        ):
            latest[key] = row
    rows = list(latest.values())
    errors: dict[str, list[float]] = {}
    for row in rows:
        for metric, value in (row.get("absolute_errors") or {}).items():
            errors.setdefault(metric, []).append(float(value))
    return {
        "scope": "latest pre-kickoff role-aware projection per player-game",
        "authority": "shadow_only",
        "players_graded": len(rows),
        "pending_player_snapshots": sum(
            row.get("status") == "pending"
            for row in payload.get("player_snapshots", [])
        ),
        "mae": {
            metric: round(statistics.fmean(values), 4)
            for metric, values in sorted(errors.items()) if values
        },
        "anytime_td_brier": _mean(rows, "anytime_td_brier"),
    }


def _units(result: str, price) -> float:
    if result == "win":
        price = float(price or -110)
        return round(price / 100.0 if price > 0 else 100.0 / -price, 3)
    return -1.0 if result == "loss" else 0.0


def _grade_bet(bet: dict, results: dict, player_index: dict, published: set) -> None:
    line = float(bet["line"])
    result = results.get((bet["season"], bet["week"], bet["home"], bet["away"]))
    if result is None:
        return
    if bet["family"] in ("spread", "total"):
        margin = float(result["home_score"]) - float(result["away_score"])
        total = float(result["home_score"]) + float(result["away_score"])
        if bet["family"] == "spread":
            value = (margin if bet["side"] == "home" else -margin) + line
            bet["actual"] = margin
        else:
            value = (total - line) if bet["side"] == "over" else (line - total)
            bet["actual"] = total
    else:
        team = bet.get("team") or ""
        if (bet["season"], bet["week"], team) not in published:
            return                                  # box score not out yet
        row = player_index.get((bet["season"], bet["week"], team, bet.get("player_id") or ""))
        if row is None:
            bet.update({"status": "graded", "graded_at": _stamp(), "result": "void",
                        "units": 0.0, "actual": None})   # books void a DNP
            return
        actual = _actual_player_metrics(row, {bet["metric"]}).get(bet["metric"])
        if actual is None:
            return
        value = (actual - line) if bet["side"] == "over" else (line - actual)
        bet["actual"] = actual
    outcome = "push" if abs(value) < 1e-9 else "win" if value > 0 else "loss"
    bet.update({"status": "graded", "graded_at": _stamp(), "result": outcome,
                "units": _units(outcome, bet.get("price"))})


def _closing_lines(snapshots: list[dict]) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for row in snapshots:
        k = (row.get("season"), row.get("week"), row.get("home"), row.get("away"))
        if k not in out or row.get("recorded_at", "") > out[k].get("recorded_at", ""):
            out[k] = row
    return out


def _clv(spot: dict, close: dict | None) -> float | None:
    """How much better the logged number was than the last pre-kickoff number."""
    if not close:
        return None
    line = float(spot["line"])
    if spot["family"] == "spread":
        if close.get("book_margin") is None:
            return None
        closing = -close["book_margin"] if spot["side"] == "home" else close["book_margin"]
        return round(line - closing, 2)
    if close.get("book_total") is None:
        return None
    closing = float(close["book_total"])
    return round((closing - line) if spot["side"] == "over" else (line - closing), 2)


def _sharp_summary(payload: dict, season: int) -> dict:
    out: dict[str, dict] = {}
    for spot in payload.get("sharp_spots", []):
        if int(spot.get("season", 0)) != season:
            continue
        fam = out.setdefault(spot["family"], {"win": 0, "loss": 0, "push": 0, "void": 0,
                                              "pending": 0, "units": 0.0, "clv": []})
        if spot.get("status") != "graded":
            fam["pending"] += 1
            continue
        fam[spot["result"]] += 1
        fam["units"] = round(fam["units"] + float(spot.get("units") or 0.0), 3)
        if spot.get("clv") is not None:
            fam["clv"].append(float(spot["clv"]))
    for fam in out.values():
        values = fam.pop("clv")
        fam["mean_clv"] = round(statistics.fmean(values), 2) if values else None
    return out


def _best_bet_summary(payload: dict, season: int) -> dict:
    out: dict[str, dict] = {}
    for bet in payload.get("best_bets", []):
        if int(bet.get("season", 0)) != season:
            continue
        family = out.setdefault(bet["family"], {"win": 0, "loss": 0, "push": 0, "void": 0,
                                                "pending": 0, "units": 0.0})
        if bet.get("status") != "graded":
            family["pending"] += 1
            continue
        family[bet["result"]] += 1
        family["units"] = round(family["units"] + float(bet.get("units") or 0.0), 3)
        if bet.get("clv") is not None:
            family.setdefault("clv", []).append(float(bet["clv"]))
    for family in out.values():
        values = family.pop("clv", [])
        family["mean_clv"] = round(statistics.fmean(values), 2) if values else None
    return out


def _record_best_bets(payload: dict, best_bets: list[dict] | None, now: datetime,
                      key: str = "best_bets") -> None:
    """Log each pick the first time it is published, and lock it there.

    A reader acts on a pick when it appears, at the number it shows then, so the
    first published side, line and price are what is graded: later builds neither
    rewrite it at a moved line nor withdraw it when it drops off the list.
    """
    if best_bets is None:
        return
    known = {bet["pick_id"] for bet in payload[key]}
    for bet in best_bets:
        kickoff = _parse(bet.get("kickoff"))
        if kickoff is None or kickoff <= now or bet["pick_id"] in known:
            continue
        payload[key].append({**bet, "recorded_at": _stamp(now), "status": "pending",
                             "authority": "shadow_only"})
        known.add(bet["pick_id"])

def summary(payload: dict, *, season: int) -> dict:
    graded = [
        row for row in payload.get("snapshots", [])
        if row.get("status") == "graded" and int(row.get("season", 0)) == season
    ]
    latest: dict[tuple, dict] = {}
    for row in graded:
        key = (row["season"], row["week"], row["home"], row["away"])
        if key not in latest or row.get("recorded_at", "") > latest[key].get("recorded_at", ""):
            latest[key] = row
    rows = list(latest.values())
    ats = [row.get("ats_result") for row in rows if row.get("ats_result")]
    totals = [row.get("total_result") for row in rows if row.get("total_result")]
    return {
        "scope": "latest pre-kickoff DraftKings snapshot per game",
        "authority": "shadow_only",
        "games_graded": len(rows),
        "pending_snapshots": sum(
            row.get("status") == "pending" for row in payload.get("snapshots", [])
        ),
        "model_mae": _mean(rows, "model_abs_error"),
        "consensus_mae": _mean(rows, "consensus_abs_error"),
        "book_mae": _mean(rows, "book_abs_error"),
        "ats": {name: ats.count(name) for name in ("win", "loss", "push")},
        "model_total_mae": _mean(rows, "model_total_abs_error"),
        "book_total_mae": _mean(rows, "book_total_abs_error"),
        "totals": {name: totals.count(name) for name in ("win", "loss", "push")},
        "players": _player_summary(payload, season),
        "best_bets": _best_bet_summary(payload, season),
        "sharp_spots": _sharp_summary(payload, season),
    }


_PLAY_FIELDS = (
    "season", "week", "away", "home", "kickoff", "recorded_at", "book", "book_margin",
    "book_total", "model_margin", "model_total", "status", "actual_margin",
    "actual_total", "ats_result", "total_result", "graded_at", "authority",
)


def plays(payload: dict, *, season: int) -> list[dict]:
    """One row per game: the last pre-kickoff snapshot, graded once final.

    This is the readable log of what the board showed and how it came out; the
    full ledger keeps every vintage.
    """
    latest: dict[tuple, dict] = {}
    for row in payload.get("snapshots", []):
        if int(row.get("season", 0)) != season:
            continue
        key = (row["season"], row["week"], row["home"], row["away"])
        if key not in latest or row.get("recorded_at", "") > latest[key].get(
            "recorded_at", ""
        ):
            latest[key] = row
    out = []
    for row in latest.values():
        play = {field: row.get(field) for field in _PLAY_FIELDS}
        if row.get("model_margin") is not None and row.get("book_margin") is not None:
            play["ats_side"] = (
                row["home"] if float(row["model_margin"]) > float(row["book_margin"])
                else row["away"]
            )
        if row.get("model_total") is not None and row.get("book_total") is not None:
            play["total_side"] = (
                "over" if float(row["model_total"]) > float(row["book_total"]) else "under"
            )
        out.append(play)
    return sorted(out, key=lambda play: (play["week"], play.get("kickoff") or "",
                                         play["away"]))


def update(
    *,
    season: int,
    projections: list["GameProjection"],
    player_projections: list["PlayerProjection"] | None = None,
    player_results: list[dict] | None = None,
    schedule: list[dict],
    best_bets: list[dict] | None = None,
    sharp_spots: list[dict] | None = None,
    path: Path = DEFAULT_PATH,
    recorded_at: datetime | None = None,
) -> dict:
    """Grade pending rows, then append every unseen pre-kickoff book vintage."""
    payload = _load(path)
    results = _result_index(schedule)
    for snapshot in payload["snapshots"]:
        if snapshot.get("status") != "pending":
            continue
        result = results.get(
            (snapshot["season"], snapshot["week"], snapshot["home"], snapshot["away"])
        )
        if result is not None:
            _grade(snapshot, result)

    player_index = _player_result_index(player_results or [])
    published = _published_team_weeks(player_results or [])
    for snapshot in payload["player_snapshots"]:
        if snapshot.get("status") != "pending":
            continue
        game_result = results.get(
            (snapshot["season"], snapshot["week"], snapshot["home"], snapshot["away"])
        )
        if game_result is None:
            continue
        if (snapshot["season"], snapshot["week"], snapshot["team"]) not in published:
            continue
        result = player_index.get((
            snapshot["season"], snapshot["week"], snapshot["team"], snapshot["player_id"]
        ))
        # Once the team's box score is published, an absent stat row is a
        # zero-stat DNP, not an indefinitely pending observation.
        _grade_player(snapshot, result)

    closing = _closing_lines(payload["snapshots"])
    for bet in payload["best_bets"]:
        if bet.get("status") == "pending":
            _grade_bet(bet, results, player_index, published)
            if bet.get("status") == "graded" and bet["family"] in ("spread", "total"):
                bet["clv"] = _clv(bet, closing.get(
                    (bet["season"], bet["week"], bet["home"], bet["away"])))
    for spot in payload["sharp_spots"]:
        if spot.get("status") == "pending":
            _grade_bet(spot, results, player_index, published)
            if spot.get("status") == "graded":
                spot["clv"] = _clv(spot, closing.get(
                    (spot["season"], spot["week"], spot["home"], spot["away"])))

    now = recorded_at or _now()
    _record_best_bets(payload, best_bets, now)
    _record_best_bets(payload, sharp_spots, now, key="sharp_spots")
    known = {row.get("snapshot_id") for row in payload["snapshots"]}
    for projection in projections:
        kickoff = _parse(projection.kickoff_utc)
        if kickoff is None or kickoff <= now or not projection.book_name:
            continue
        quote_key = projection.book_last_update or _stamp(now)
        snapshot_id = "|".join((
            str(projection.season), str(projection.week), projection.away, projection.home,
            str(projection.book_key), quote_key,
        ))
        if snapshot_id in known:
            continue
        payload["snapshots"].append({
            "snapshot_id": snapshot_id,
            "recorded_at": _stamp(now),
            "season": projection.season,
            "week": projection.week,
            "home": projection.home,
            "away": projection.away,
            "kickoff": projection.kickoff_utc,
            "model_lineage": "2026.09-time-forward-audited-symmetric-matchup",
            "model_margin": projection.model_margin,
            "consensus_margin": projection.market_margin,
            "book": projection.book_name,
            "book_key": projection.book_key,
            "book_margin": projection.book_margin,
            "book_total": projection.book_total,
            "home_moneyline": projection.home_moneyline,
            "away_moneyline": projection.away_moneyline,
            "book_last_update": projection.book_last_update,
            "model_total": projection.projected_total,
            "status": "pending",
            "authority": "shadow_only",
        })
        known.add(snapshot_id)

    known_players = {row.get("snapshot_id") for row in payload["player_snapshots"]}
    # Only the latest pre-kickoff projection per player-game is ever scored
    # (`_player_summary`), so a newer one replaces a pending older one instead of
    # piling up: every build used to add a full copy of the slate, ~8 per game.
    pending_at = {
        _player_key(row): i for i, row in enumerate(payload["player_snapshots"])
        if row.get("status") == "pending"
    }
    for projection in player_projections or []:
        kickoff = _parse(projection.kickoff_utc)
        if kickoff is None or kickoff <= now:
            continue
        snapshot_id = "|".join((
            str(projection.season), str(projection.week), projection.game_id,
            projection.player_id, projection.model_version, _stamp(now),
        ))
        if snapshot_id in known_players:
            continue
        # Home/away identity makes completion grading independent of whether the
        # player produced an official stat row.
        home = projection.team if projection.home else projection.opponent
        away = projection.opponent if projection.home else projection.team
        row = {
            "snapshot_id": snapshot_id,
            "recorded_at": _stamp(now),
            "season": projection.season,
            "week": projection.week,
            "game_id": projection.game_id,
            "home": home,
            "away": away,
            "kickoff": projection.kickoff_utc,
            "team": projection.team,
            "opponent": projection.opponent,
            "player_id": projection.player_id,
            "player_name": projection.player_name,
            "position": projection.position,
            "depth_rank": projection.depth_rank,
            "injury_status": projection.injury_status,
            "role_continuity": projection.role_continuity,
            "persistence_weight": projection.persistence_weight,
            "confidence": projection.confidence,
            "model_version": projection.model_version,
            "scheme_context": projection.scheme_context,
            "metrics": projection.metrics,
            "status": "pending",
            "authority": "shadow_only",
        }
        existing = pending_at.get(_player_key(row))
        if existing is None:
            pending_at[_player_key(row)] = len(payload["player_snapshots"])
            payload["player_snapshots"].append(row)
        else:
            payload["player_snapshots"][existing] = row
        known_players.add(snapshot_id)

    payload["schema_version"] = SCHEMA_VERSION
    payload["updated_at"] = _stamp(now)
    payload["snapshots"] = [
        row for row in payload["snapshots"] if int(row.get("season", season)) >= season - 2
    ][-5000:]
    payload["player_snapshots"] = _latest_per_player_game([
        row for row in payload["player_snapshots"]
        if int(row.get("season", season)) >= season - 2
    ])[-100000:]
    payload["summary"] = summary(payload, season=season)
    _write(path, payload)
    return payload
