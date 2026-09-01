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

COEFFICIENTS_HOME_FIELD = matrix.COEFFICIENTS["home_field"]

# Short labels for the breakdown tiles. `matrix.FEATURE_LABELS` stays descriptive
# for the methodology table, but the kernel truncates a tile label with an
# ellipsis and at the two-column board breakpoint it has about 84px to work with
# -- "First-down rate" rendered as "FIRST-DOWN RA...". These are the football
# words for the same quantities and they fit.
_TILE_LABELS = {
    "epa": "EPA/play",
    "first_down": "First downs",
    "explosive": "Explosives",
    "sack": "Sacks",
    "turnover": "Turnovers",
}

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
    """The MODEL's win probability, with the market's fair price beside it.

    Deliberately not the published probability. Spread and Total both show the
    model's number and put the market in the sub-line, so a moneyline derived
    from the published margin contradicted the tile next to it every time the
    two disagreed about who was favoured: "IND -1.5" beside "BAL 60%". Both
    numbers were correct and the card was incoherent.
    """
    if p.model_win_probability is None:
        return Tile(label="Moneyline", value="—", state="no rating", tone="mut")
    favourite = p.home if p.model_win_probability >= 0.5 else p.away
    probability = max(p.model_win_probability, 1.0 - p.model_win_probability)
    if p.market_fair_home is not None:
        # Quoted on the SAME side the model favours, not on whichever side the
        # market favours -- otherwise two percentages sit side by side describing
        # different teams.
        market = (p.market_fair_home if favourite == p.home
                  else 1.0 - p.market_fair_home)
        prices = f"{_american(p.home_moneyline)}/{_american(p.away_moneyline)}"
        state = f"market {market * 100:.0f}% · {prices}"
    else:
        state = "model only — no paired price"
    return Tile(
        label="Moneyline",
        value=f"{favourite} {probability * 100:.0f}%",
        state=state,
        tone=_ACTION_TONE.get(p.action, "mut"),
        note=("Model win probability at its 13.32-point residual SD, against the "
              "paired no-vig market on the same side. De-vigging is paired only; "
              "a one-sided fair price is guesswork."),
        priced=p.market_fair_home is not None,
    )


def _breakdown_group(p: GameProjection) -> Group | None:
    """How the model got there, family by family -- the shelf this board lacked.

    The matchup model is linear and symmetric, so the margin decomposes exactly:
    these five contributions plus the home-field term reconcile to the efficiency
    margin with no residual. A breakdown whose parts do not add up to the number
    above it is worse than none at all, so the arithmetic is exact rather than
    indicative, and `tests/test_site.py` checks that it still reconciles.
    """
    home_form, away_form = p.home_form, p.away_form
    if home_form is None or away_form is None:
        return None
    contributions = matrix.margin_contributions(home_form, away_form)
    if contributions is None:
        return None
    # Home field is a term of the same sum, so it belongs on the same shelf. Left
    # off, the tiles reconciled on a hosted game and silently did NOT on a neutral
    # one, while the note claimed they always did -- the Melbourne game was the
    # counter-example sitting on the board. Including it makes the shelf
    # self-evidently complete at both venues.
    if not p.neutral:
        contributions = {**contributions, "home_field": COEFFICIENTS_HOME_FIELD}
    tiles = []
    # Largest effect first: a reader scanning one card wants the reason, not an
    # alphabetical list of factors.
    for stat, points in sorted(contributions.items(), key=lambda kv: -abs(kv[1])):
        beneficiary = p.home if points > 0 else p.away
        if stat == "home_field":
            tiles.append(Tile(
                label="Home field",
                value=f"{beneficiary} {abs(points):.1f}",
                state="fitted home advantage",
                tone="mut",
                note=("The measured home-field term of the matchup model. Every tile on "
                      "this shelf is a term of one sum, and together they equal the "
                      "efficiency margin exactly."),
            ))
            continue
        # NET, not raw offence. The contribution is g x (net_home - net_away)
        # where net = what a team's offence produces minus what its defence
        # allows, so the two numbers shown must be the nets or the sub-line
        # contradicts the tile above it. New England out-produces Seattle on
        # offensive EPA and still loses the family, because Seattle's defence
        # allows far less -- printing only the offence made that look like a
        # sign error.
        def net(form) -> float:
            return getattr(form, f"off_{stat}") - getattr(form, f"def_{stat}")

        away_net, home_net = net(away_form), net(home_form)
        shown = (f"{away_net:+.3f} / {home_net:+.3f}" if stat == "epa"
                 else f"{away_net * 100:+.1f} / {home_net * 100:+.1f} pp")
        tiles.append(Tile(
            label=_TILE_LABELS[stat],
            value=f"{beneficiary} {abs(points):.1f}",
            state=f"net {p.away} {shown.split(' / ')[0]} / {p.home} {shown.split(' / ')[1]}",
            tone="mut",
            note=(f"Points of home margin from the {matrix.FEATURE_LABELS[stat].lower()} "
                  f"matchup. Net is a team's own rate minus the rate its defence "
                  f"allows, and the contribution is the coefficient times the "
                  f"difference of the two nets. Every tile on this shelf is a term "
                  f"of one sum, and together they equal the efficiency margin."),
        ))
    return Group(label="How the model got there", tiles=tuple(tiles), tag="breakdown",
                 market=False)


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
            value=f"{leader} +{abs(difference):.1f}",
            state=f"{p.away} {away_value:+.1f} / {p.home} {home_value:+.1f}",
            tone="mut",
            note=("Points per game above an average unit, from the opponent-adjusted "
                  "matchup model. Offence plus defence is the team's efficiency rating."),
        ))
    if p.rating_margin is not None:
        # Same "{team} +{amount}" shape as the two tiles above it. A bare signed
        # number in the third slot made the reader work out whose sign it was.
        leader = p.home if p.rating_margin >= 0 else p.away
        tiles.append(Tile(
            label="Rating gap",
            value=f"{leader} +{abs(p.rating_margin):.1f}",
            state=("power ratings · neutral site" if p.neutral
                   else "power ratings + home field"),
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
               coaches: tuple[str, str] = ("", ""),
               records: dict | None = None,
               ratings_table: dict | None = None,
               row: dict | None = None) -> Card:
    """One projection -> one board card."""
    records = records or {}
    ratings_table = ratings_table or {}
    row = row or {}

    def side(abbr: str, score: float | None, opponent_score: float | None) -> Side:
        # Rating and record, not the moneyline: the price is already on its own
        # tile, and a bare "+150" beside a team name tells a reader nothing about
        # whether the team is good.
        parts = []
        rating = ratings_table.get(abbr)
        if rating is not None:
            parts.append(f"{rating:+.1f}")
        record = records.get(abbr)
        if record is not None and record.played:
            parts.append(f"{record.label} '{record.season % 100:02d}")
        return Side(
            abbr=abbr,
            score=f"{score:.0f}" if score is not None else "—",
            detail=" · ".join(parts),
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
    for build in (_matchup_group, _breakdown_group):
        shelf = build(p)
        if shelf is not None:
            groups.append(shelf)

    # Everything a reader needs to place the game before reading a number.
    context = [p.kickoff]
    if p.neutral:
        context.append("neutral site")
    if row.get("div_game"):
        context.append("division")
    roof = str(row.get("roof") or "")
    if roof in {"dome", "closed"}:
        context.append("indoors")
    elif roof == "retractable":
        context.append("retractable roof")
    rest = row.get("home_rest")
    if rest is not None and float(rest) > 7:
        context.append(f"{int(float(rest))}d rest {p.home}")
    return Card(
        key=key or f"{p.away}@{p.home}",
        league="NFL",
        start_text=" · ".join(part for part in context if part),
        status_label=p.action,
        status_tone=_ACTION_TONE.get(p.action, "mut"),
        away=side(p.away, p.projected_away_score, p.projected_home_score),
        home=side(p.home, p.projected_home_score, p.projected_away_score),
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
            records={team: slate.record_for(team)
                     for team in (projection.home, projection.away)
                     if slate.record_for(team) is not None},
            ratings_table=slate.table,
            row=row,
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
        # Labels carry no count: the kernel appends its own match count to every
        # filter, so "All 16" rendered as "ALL 16 16".
        filters=[("all", "All"), ("fullgame", "Priced"), ("breakdown", "With form")],
        sorts=[("start", "Kickoff"), ("picks", "Priced markets")],
        empty_text=(
            "No regular-season games found for this week. The schedule comes from "
            "nflverse; a missing week is a source problem, not an empty slate."
        ),
    )
