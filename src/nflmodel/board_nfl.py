"""NFL adapter for the shared Board kernel (``board.py``).

The kernel is vendored byte-identical from mlb-model and owns the card anatomy,
filters, counters and empty states. This module owns the only football-specific
decisions.

* **scores** are the model's projected points, split out of its total and margin
  exactly as mlb-model splits expected runs. Earlier this slot held a fair win
  probability, which is not a score and left the card's most prominent number
  saying something the eye reads as one.
* **principals** are the starting quarterbacks when nflverse has published them
  and the head coaches when it has not, with the label saying which. A card with
  an empty principal row looks like a failed image load; inventing a name would
  be worse than either.
* **groups** are Full Game -- spread, total and moneyline -- plus an unpriced
  matchup shelf that shows *why* the projection landed where it did.
* **every tile carries the action its authority permits.** At `SPREAD_LAMBDA = 0`
  the published margin is the market, so a card showing a number without its
  permission would be a calibrated price dressed as an edge. `MONITOR` is the
  ceiling today; `BET` is unreachable while the authority is RESEARCH_ONLY.

**On the "Picks" counter.** The kernel counts priced markets and labels the
result Picks. These markets *are* priced -- that count is accurate -- but a
priced market is not a recommendation, and the board's own copy says so rather
than leaving the word to imply otherwise. Gems are always zero: a gem asserts an
actionable edge and this authority permits none.
"""
from __future__ import annotations

from . import authority as auth
from . import matrix, teams
from .board import Board, Card, Group, Principal, Side, Tile
from .forecast import GameProjection

# The gate maps an action to how a tile reads. BET is unreachable while the
# authority is RESEARCH_ONLY; it is listed so the mapping stays total if a
# challenger ever promotes.
_ACTION_TONE = {
    "BET": "pos",
    "MONITOR": "side",
    "REVIEW": "warnc",
    "AVOID": "mut",
}


def _logo(abbr: str) -> str:
    team = teams.get(abbr)
    return (f'<img src="{team.logo}" alt="" loading="lazy" width="34" height="34" '
            f'style="border-radius:6px">')


def _handicap(margin: float | None, home: str, away: str) -> str:
    """A margin as a bettor reads it: the favourite and its number."""
    if margin is None:
        return "—"
    if abs(margin) < 0.05:
        return "PK"
    favourite = home if margin > 0 else away
    return f"{favourite} {-abs(margin):.1f}"


def _american(value) -> str:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return "—"
    return f"{number:+d}"


def _spread_tile(p: GameProjection) -> Tile:
    if p.model_margin is None:
        return Tile(label="Spread", value="—", state="no rating", tone="mut")
    state = "model only — no market price"
    if p.market_margin is not None:
        state = (f"market {_handicap(p.market_margin, p.home, p.away)} · "
                 f"gap {p.market_gap:+.1f}")
    return Tile(
        label="Spread",
        value=_handicap(p.model_margin, p.home, p.away),
        state=state,
        tone=_ACTION_TONE.get(p.action, "mut"),
        note=("Model margin against the closing line. The published margin equals the "
              "market at lambda 0, so the difference is an information gap, not an edge."),
        priced=p.market_margin is not None,
    )


def _total_tile(p: GameProjection) -> Tile:
    if p.projected_total is None:
        return Tile(label="Total", value="—", state="no form", tone="mut")
    star = "" if p.total_modelled else "*"
    if p.market_total is not None:
        state = f"market {p.market_total:.1f} · gap {p.total_gap:+.1f}"
    elif not p.total_modelled:
        state = "league mean — no form"
    else:
        state = "model only — no market price"
    return Tile(
        label="Total",
        value=f"{p.projected_total:.1f}{star}",
        state=state,
        tone=_ACTION_TONE.get(p.action, "mut"),
        note=("Model total, shrunk 30% toward the league mean because the raw estimate is "
              "over-dispersed. Residual SD is 13.7 points — read it as a centre of mass."),
        priced=p.market_total is not None,
    )


def _moneyline_tile(p: GameProjection) -> Tile:
    if p.win_probability is None:
        return Tile(label="Moneyline", value="—", state="no rating", tone="mut")
    favourite = p.home if p.win_probability >= 0.5 else p.away
    probability = max(p.win_probability, 1.0 - p.win_probability)
    if p.market_fair_home is not None:
        market = max(p.market_fair_home, 1.0 - p.market_fair_home)
        prices = f"{_american(p.home_moneyline)}/{_american(p.away_moneyline)}"
        state = f"market {market * 100:.0f}% · {prices}"
    else:
        state = "model only — no paired price"
    return Tile(
        label="Moneyline",
        value=f"{favourite} {probability * 100:.0f}%",
        state=state,
        tone=_ACTION_TONE.get(p.action, "mut"),
        note=("Win probability from the published margin at the model's 13.32-point "
              "residual SD. De-vigging is paired only; a one-sided fair price is guesswork."),
        priced=p.market_fair_home is not None,
    )


def _matchup_group(p: GameProjection) -> Group | None:
    """Unpriced shelf: which unit is carrying the projection.

    Marked ``market=False`` so the kernel does not label it "no price", which on
    an explanatory shelf reads as a missing feed rather than as context.
    """
    home_form, away_form = p.home_form, p.away_form
    if home_form is None or away_form is None:
        return None
    tiles: list[Tile] = []
    for label, index in (("Offense edge", matrix.offense_index),
                         ("Defense edge", matrix.defense_index)):
        home_value, away_value = index(home_form), index(away_form)
        if home_value is None or away_value is None:
            continue
        difference = home_value - away_value
        leader = p.home if difference > 0 else p.away
        tiles.append(Tile(
            label=label,
            value=f"{abs(difference):.1f}",
            state=f"{leader} · {p.away} {away_value:+.1f} / {p.home} {home_value:+.1f}",
            tone="mut",
            note=("Points per game above an average unit, from the opponent-adjusted "
                  "matchup model. Offence plus defence is the team's efficiency rating."),
        ))
    if p.rating_margin is not None:
        tiles.append(Tile(
            label="Rating gap",
            value=f"{p.rating_margin:+.1f}",
            state="power ratings + home field",
            tone="mut",
            note=("The other half of the published margin: opponent-adjusted scoring "
                  "margin, home field removed before rating and added back here."),
        ))
    if not tiles:
        return None
    return Group(label="Why this projection", tiles=tuple(tiles), tag="matchup",
                 market=False)


def _efficiency_stat(form) -> str:
    """The team's efficiency rating, labelled as the team's rather than the
    person's. A bare "+4.4" next to a coach's name reads as a rating OF the
    coach, which is not what this measures. Zero is normalised because Python
    renders a tiny negative as "-0.0", which looks like a defect.
    """
    if form is None:
        return ""
    value = matrix.efficiency_rating(form)
    if value is None:
        return ""
    if abs(value) < 0.05:
        value = 0.0
    return f"team {value:+.1f}"


def _headline(p: GameProjection) -> str:
    if p.model_margin is None:
        return "No rating for this matchup"
    if p.market_margin is None:
        return f"Model: {_handicap(p.model_margin, p.home, p.away)}, total {p.projected_total:.1f}"
    if abs(p.market_gap) < 0.5:
        return "Model agrees with the closing line"
    side = p.home if p.market_gap > 0 else p.away
    return f"Model is {abs(p.market_gap):.1f} points on {side} — a gap, not an edge"


def build_card(p: GameProjection, *, key: str = "",
               qbs: tuple[str, str] = ("", ""),
               coaches: tuple[str, str] = ("", "")) -> Card:
    """One projection -> one board card."""
    def side(abbr: str, score: float | None, opponent_score: float | None,
             moneyline) -> Side:
        return Side(
            abbr=abbr,
            score=f"{score:.0f}" if score is not None else "—",
            detail=_american(moneyline) if moneyline is not None else "",
            logo_html=_logo(abbr),
            favored=(score is not None and opponent_score is not None
                     and score > opponent_score),
        )

    away_qb, home_qb = qbs
    away_coach, home_coach = coaches
    if away_qb and home_qb:
        principal_label, names = "Starting quarterbacks", ((away_qb, p.away), (home_qb, p.home))
    elif away_coach and home_coach:
        principal_label, names = "Head coaches", ((away_coach, p.away), (home_coach, p.home))
    else:
        principal_label, names = "", ()
    principals = tuple(
        Principal(name=name, team=team, stat=_efficiency_stat(form))
        for (name, team), form in zip(names, (p.away_form, p.home_form))
    )

    groups = [Group(label="Full game",
                    tiles=(_spread_tile(p), _total_tile(p), _moneyline_tile(p)),
                    tag="fullgame", state=p.action)]
    matchup = _matchup_group(p)
    if matchup is not None:
        groups.append(matchup)

    venue = " · neutral site" if p.neutral else ""
    return Card(
        key=key or f"{p.away}@{p.home}",
        league="NFL",
        start_text=f"{p.kickoff}{venue}",
        status_label=p.action,
        status_tone=_ACTION_TONE.get(p.action, "mut"),
        away=side(p.away, p.projected_away_score, p.projected_home_score, p.away_moneyline),
        home=side(p.home, p.projected_home_score, p.projected_away_score, p.home_moneyline),
        headline=_headline(p),
        headline_tone="mut",
        principals=principals,
        principal_label=principal_label,
        groups=tuple(groups),
        action_label="Gate",
        action_js="location.hash='authority'",
        footer_label="Why this is research only",
        footer_js="location.hash='authority'",
        note="No priced market for this game — reported rather than dropped.",
    )


def build_board(slate, *, rows_by_key: dict | None = None) -> Board:
    """Assemble the NFL board from a `season.Slate`."""
    rows_by_key = rows_by_key or {}
    cards = []
    for projection in slate.projections:
        row = rows_by_key.get(f"{projection.away}@{projection.home}", {})
        cards.append(build_card(
            projection,
            key=str(row.get("game_id") or f"{projection.away}@{projection.home}"),
            qbs=(row.get("away_qb_name", ""), row.get("home_qb_name", "")),
            coaches=(row.get("away_coach", ""), row.get("home_coach", "")),
        ))

    authority = slate.authority
    priced = sum(1 for p in slate.projections if p.market_margin is not None)
    meta = [
        f"{len(cards)} games",
        f"{priced} with a market price",
        f"authority {authority.level.value}",
        f"{len(authority.unmet_gates)} of {len(auth.REQUIRED_GATES)} production gates unmet",
        "lambda 0.000",
    ]
    return Board(
        sport="NFL",
        cards=cards,
        date_label=f"{slate.season} · Week {slate.week}",
        meta=meta,
        filters=[("all", f"All {len(cards)}"), ("fullgame", "Priced"),
                 ("matchup", "With form")],
        sorts=[("start", "Kickoff"), ("picks", "Priced markets")],
        empty_text=(
            "No regular-season games found for this week. The schedule comes from "
            "nflverse; a missing week is a source problem, not an empty slate."
        ),
    )
