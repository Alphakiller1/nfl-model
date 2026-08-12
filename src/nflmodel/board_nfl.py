"""
NFL adapter for the shared Board kernel (``board.py``).

The kernel is vendored byte-identical from mlb-model and owns the card anatomy, filters,
counters and empty states. This module owns the only football-specific decisions — and one
of them is unusual enough to state plainly:

**The authority gate is rendered as a first-class part of every card.** At `lam = 0` this
forecast equals the paired no-vig market by construction, so a card that displayed a number
without its permission would be showing a calibrated price dressed as an edge. Every tile
therefore carries the action the authority actually permits (`MONITOR` today, never `BET`),
and the board header states the level and the unmet-gate count.

* **principals** are empty — this repo has no player-level feed, and an invented one would
  be worse than an honest omission
* **groups** are Full Game only
* **scores** are the model's fair win probability, not projected points, because a
  market-anchored moneyline model does not produce a score
"""
from __future__ import annotations

from . import authority as auth
from .board import Board, Card, Group, Side, Tile

# The gate maps an action to how the tile reads. BET is unreachable while the authority is
# RESEARCH_ONLY; it is listed so the mapping stays total if a challenger ever promotes.
_ACTION_TONE = {
    "BET": "pos",
    "MONITOR": "side",
    "REVIEW": "warnc",
    "AVOID": "mut",
}


def _american(value) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:+d}"


def build_card(row: dict) -> Card:
    """One forecast row from `forecast_slate` -> one board card."""
    home, away = str(row.get("home_team", "")), str(row.get("away_team", ""))
    home_fair = row.get("home_fair")
    away_fair = row.get("away_fair")
    action = str(row.get("action") or "AVOID")
    edge = row.get("edge_vs_market")
    priced = home_fair is not None and action != "AVOID"

    def side(abbr: str, fair, american) -> Side:
        return Side(
            abbr=abbr,
            score=f"{float(fair) * 100:.0f}%" if fair is not None else "—",
            detail=f"fair {_american(american)}",
            favored=fair is not None and float(fair) >= 0.5,
        )

    if edge is None:
        value, state = "—", "no paired price"
    else:
        value = f"{float(edge) * 100:+.1f}"
        state = f"vs market · {action}"

    tile = Tile(
        label="Moneyline",
        value=value,
        state=state,
        tone=_ACTION_TONE.get(action, "mut"),
        note=(
            "Edge is measured against the paired no-vig market. At lam = 0 the forecast "
            "equals that market by construction, so a 0.0 edge is the expected result."
        ),
        # Never a gem: a gem asserts an actionable edge, and this authority permits none.
        gem=False,
        priced=priced,
    )

    return Card(
        key=str(row.get("game") or f"{away}@{home}"),
        league="NFL",
        status_label=action,
        status_tone=_ACTION_TONE.get(action, "mut"),
        away=side(away, away_fair, row.get("away_american")),
        home=side(home, home_fair, row.get("home_american")),
        headline=(
            "Matches the market — calibrated price, not an edge"
            if edge is not None and abs(float(edge)) < 1e-9
            else f"Model differs from market by {value}pt"
        ),
        headline_tone="mut" if action in {"AVOID", "MONITOR"} else "side",
        principals=(),
        groups=(
            Group(
                label="Full game",
                tiles=(tile,),
                tag="fullgame",
                state=action,
            ),
        ),
        action_label="Gate",
        action_js="location.hash='authority'",
        footer_label="Why this is research only",
        footer_js="location.hash='authority'",
        note="No paired price for this game — reported rather than dropped.",
    )


def build_board(payload: dict) -> Board:
    """Assemble the NFL board from a `forecast_slate` payload."""
    games = payload.get("games") or []
    skipped = payload.get("skipped") or []
    cards = [build_card(row) for row in games]
    cards += [
        Card(
            key=str(row.get("game") or "?"),
            league="NFL",
            status_label="AVOID",
            status_tone="mut",
            away=Side(abbr=str(row.get("game", "?")).split("@")[0] or "?"),
            home=Side(abbr=str(row.get("game", "?")).split("@")[-1] or "?"),
            headline="No usable paired price",
            headline_tone="mut",
            note=str(row.get("reason") or "Missing or unpairable quote."),
        )
        for row in skipped
    ]

    level = str(payload.get("authority") or auth.Level.RESEARCH_ONLY.value)
    unmet = payload.get("unmet_gates") or []
    lam = payload.get("lam")

    meta = [
        f"{len(cards)} games",
        f"authority {level}",
        f"{len(unmet)} of {len(auth.REQUIRED_GATES)} production gates unmet",
    ]
    if lam is not None:
        meta.append(f"lam {float(lam):.3f}")

    return Board(
        sport="NFL",
        cards=cards,
        meta=meta,
        filters=[("all", f"All {len(cards)}"), ("fullgame", "Priced")],
        sorts=[("start", "Matchup"), ("picks", "Priced markets")],
        empty_text=(
            "No slate loaded. Point `nfl-model build-site --games slate.json` at a slate "
            "with paired prices and the board fills in."
        ),
    )
